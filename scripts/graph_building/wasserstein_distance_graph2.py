"""
Build dense parcel-to-parcel similarity graphs for one subject.

Three edge definitions are available via --method (comma-separated):
    was   Wasserstein-1, approximated by mean absolute quantile difference
          over a common 15-point quantile grid.
    kl    Symmetrized Gaussian KL divergence between parcel (mean, variance)
          summaries: 0.5 * (KL(P||Q) + KL(Q||P)).
    meiq  Robust distance between parcel (median, IQR) summaries:
          |median_i - median_j| / (0.5 * (IQR_i + IQR_j)).

Every method's distance is converted to a similarity with the same
--sim-formula transform, and weighted degree is always computed and stored
as a float64 memmap. The dense (N, N) adjacency matrix is only written to
disk when --save-matrix true is passed — at scale (up to ~90k parcels) the
full matrix is far larger than the degree vector, so it is skipped by
default. Compute cost still grows quadratically with the parcel count either
way, since every pairwise similarity is computed to derive weighted degree;
only the disk-write cost is avoided.

Usage:
    python scripts/graph_building/wasserstein_distance_graph2.py (--input-folder PATH | --subject-id ID) --method was,kl,meiq --sim-formula {1,2} [options]

Parameters:
    --input-folder PATH      Direct subject parcel-vector folder.
    --input-root PATH        Subject-folder root (default: outputs/jacobian_parcel_vectors).
    --subject-id TEXT        Folder name under --input-root when --input-folder is omitted.
    --output-folder PATH     Custom graph output folder. Only valid with a single --method.
    --method LIST            Required. Comma-separated subset of {was, kl, meiq}.
    --sim-formula {1,2}      Required transform: 1 = exp(-W), 2 = 1/(1+W).
    --save-matrix {true,false}  Write the dense adjacency matrix to disk (default: false).
    --num-workers INT        Worker processes (default: 8).
    --block-size INT         Rows per worker job (default: 500).
    --progress-every INT     Progress interval in completed blocks (default: 10).

Outputs (per method, under its own output folder/<subject_id>/):
    weighted_degree.dat, metadata.npy, metadata.json, parcel_order.txt, and
    adjacency_matrix.dat only when --save-matrix true.

Examples:
    python scripts/graph_building/wasserstein_distance_graph2.py --subject-id sub-0091 --method was --sim-formula 1
    python scripts/graph_building/wasserstein_distance_graph2.py --subject-id sub-0091 --method was,kl,meiq --sim-formula 2 --num-workers 4
    python scripts/graph_building/wasserstein_distance_graph2.py --input-folder outputs/jacobian_parcel_vectors/sub-OAS30999 --method kl --sim-formula 1
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

METHOD_WASSERSTEIN = "was"
METHOD_KL = "kl"
METHOD_MEDIAN_IQR = "meiq"
METHOD_CHOICES = (METHOD_WASSERSTEIN, METHOD_KL, METHOD_MEDIAN_IQR)
METHOD_LABELS = {
    METHOD_WASSERSTEIN: "Wasserstein-1 (mean absolute quantile difference)",
    METHOD_KL: "Symmetrized Gaussian KL divergence (mean/variance)",
    METHOD_MEDIAN_IQR: "Robust median/IQR distance",
}
METHOD_NODE_FEATURE_LABELS = {
    METHOD_WASSERSTEIN: "15-point quantile representation of parcel log-Jacobian values",
    METHOD_KL: "parcel log-Jacobian mean and variance (Gaussian summary)",
    METHOD_MEDIAN_IQR: "parcel log-Jacobian median and IQR (robust summary)",
}
# Legacy folder name kept for "was" so existing consumers (validate_cohort_outputs.py,
# parcel_statistics.py, the notebooks) don't need to change; kl/meiq are new.
METHOD_OUTPUT_PREFIX = {
    METHOD_WASSERSTEIN: "wasserstein_graphs",
    METHOD_KL: "kl_graphs",
    METHOD_MEDIAN_IQR: "median_iqr_graphs",
}

# Epsilon floors for degenerate (near-constant) parcels, to avoid divide-by-zero
# and log(0) in the KL and median/IQR distance formulas.
MIN_VARIANCE = 1e-6
MIN_IQR = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load parcel vectors once, then build one or more dense similarity "
            "graphs (Wasserstein / KL / median-IQR) from them."
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
            "Directory where graph outputs will be saved. Only valid when --method "
            "names exactly one method; otherwise each method uses its own default folder."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help=(
            "Required, comma-separated subset of {was, kl, meiq} to build. "
            "Example: --method was,kl,meiq"
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
            "Required distance-to-similarity transform, applied to every requested "
            "method. Use 1 for exp(-W) or 2 for 1/(1+W)."
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


def parse_methods(raw: str) -> list[str]:
    methods: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in METHOD_CHOICES:
            raise ValueError(f"Unsupported --method value: {token!r}; choose from {METHOD_CHOICES}")
        if token not in methods:
            methods.append(token)
    if not methods:
        raise ValueError("--method must name at least one of: was, kl, meiq")
    return methods


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


def default_output_root(method: str, sim_formula: int, subject_id: str) -> Path:
    formula_name = SIM_FORMULA_OUTPUT_NAMES[sim_formula]
    return PROJECT_ROOT / "outputs" / f"{METHOD_OUTPUT_PREFIX[method]}_{formula_name}" / subject_id


def _to_quantile_grid(v: np.ndarray) -> np.ndarray:
    """Sort v and interpolate to QUANTILE_LEN evenly-spaced quantile levels."""
    v_sorted = np.sort(v.astype(np.float32))
    L = len(v_sorted)
    if L == QUANTILE_LEN:
        return v_sorted

    # Interpolate to the target quantile levels.
    src_q = (np.arange(L, dtype=np.float32) + 0.5) / L
    tgt_q = (np.arange(QUANTILE_LEN, dtype=np.float32) + 0.5) / QUANTILE_LEN
    return np.interp(tgt_q, src_q, v_sorted).astype(np.float32)


def load_subject_parcels(subject_folder: Path) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """
    Return parcel IDs, a float32 sorted quantile matrix of shape (N, QUANTILE_LEN)
    for the Wasserstein method, and a dict of per-parcel summary arrays
    ("mean", "var", "median", "iqr") for the KL and median/IQR methods.
    """
    order_path = subject_folder / "parcel_order.txt"
    data_path = subject_folder / "parcel_vectors.dat"
    offsets_path = subject_folder / "parcel_offsets.dat"
    if not order_path.is_file() or not data_path.is_file() or not offsets_path.is_file():
        raise ValueError(f"Missing parcel_order.txt or parcel vector .dat files under {subject_folder}")

    parcel_ids = order_path.read_text(encoding="utf-8").splitlines()
    offsets = np.fromfile(offsets_path, dtype=np.int64)
    data = np.memmap(data_path, dtype=np.float32, mode="r")
    if len(offsets) != len(parcel_ids) + 1 or offsets[0] != 0 or offsets[-1] != data.size:
        raise ValueError(f"Parcel offsets do not match parcel order/data under {subject_folder}")
    if np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError(f"Parcel offsets are not strictly increasing under {subject_folder}")

    rows: list[np.ndarray] = []
    means: list[float] = []
    variances: list[float] = []
    medians: list[float] = []
    iqrs: list[float] = []

    # Each parcel is the slice data[offsets[i]:offsets[i + 1]].
    for parcel_id, start, end in zip(parcel_ids, offsets[:-1], offsets[1:]):
        v = data[start:end]
        if not np.isfinite(v).all():
            raise ValueError(f"Parcel vector contains non-finite values: {parcel_id}")

        rows.append(_to_quantile_grid(v))
        means.append(float(np.mean(v, dtype=np.float64)))
        variances.append(max(float(np.var(v, dtype=np.float64)), MIN_VARIANCE))
        q25, q75 = np.percentile(v, [25, 75])
        medians.append(float(np.median(v)))
        iqrs.append(max(float(q75 - q25), MIN_IQR))

    if not rows:
        raise ValueError(f"No parcel vectors found under {subject_folder}")
    if len(set(parcel_ids)) != len(parcel_ids):
        raise ValueError(f"Duplicate parcel IDs found under {subject_folder}")

    # Stack the quantile rows into a single (N, QUANTILE_LEN) array for Wasserstein method
    stats = {
        "mean": np.asarray(means, dtype=np.float64),
        "var": np.asarray(variances, dtype=np.float64),
        "median": np.asarray(medians, dtype=np.float64),
        "iqr": np.asarray(iqrs, dtype=np.float64),
    }
    return parcel_ids, np.stack(rows, axis=0), stats  # (N, QUANTILE_LEN)


def _distance_block_to_similarity(
    dist: np.ndarray,
    i_start: int,
    i_end: int,
    sim_formula: int,
) -> np.ndarray:
    """Shared tail: distance -> similarity transform, sanity checks, exact diagonal."""
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

    for local_i, global_i in enumerate(range(i_start, i_end)):
        sim[local_i, global_i] = 1.0

    return sim

def _compute_block(
    i_start: int,
    i_end: int,
    sorted_matrix: np.ndarray,
    sim_formula: int,
) -> tuple[int, int, np.ndarray]:
    """
    Compute the Wasserstein similarity sub-matrix for rows [i_start, i_end).

    W1(i, j) = mean_k |sorted[i, k] - sorted[j, k]|   (exact for equal-length samples)
    similarity = exp(-W1) or 1 / (1 + W1), depending on --sim-formula

    Returns (i_start, i_end, sim_block) where sim_block has shape (i_end-i_start, N).
    """
    # Select a block of rows from the sorted quantile matrix to process
    block = sorted_matrix[i_start:i_end]

    # N is the total number of parcels and L is the number of quantiles (QUANTILE_LEN)
    N, L = sorted_matrix.shape

    # Create distance matrix of shape (B, N) where B = i_end - i_start
    dist = np.zeros((len(block), N), dtype=np.float32)

    # Iterate over each quantile
    for k in range(L):
        # Compute the absolute difference between the k-th quantile of the block and all parcels
        dist += np.abs(block[:, k : k + 1] - sorted_matrix[:, k])   # (B, N)
    dist /= L

    # Convert distances to similarities using the specified formula
    sim = _distance_block_to_similarity(dist, i_start, i_end, sim_formula)
    return i_start, i_end, sim


def _compute_block_kl(
    i_start: int,
    i_end: int,
    mean: np.ndarray,
    var: np.ndarray,
    sim_formula: int,
) -> tuple[int, int, np.ndarray]:
    """
    Symmetrized Gaussian KL divergence for rows [i_start, i_end):
    0.5 * (KL(P_i || P_j) + KL(P_j || P_i)), where P_k = N(mean[k], var[k]).
    """
    mu1 = mean[i_start:i_end, None]
    mu2 = mean[None, :]
    s1 = var[i_start:i_end, None]
    s2 = var[None, :]

    kl_pq = 0.5 * np.log(s2 / s1) + (s1 + (mu1 - mu2) ** 2) / (2 * s2) - 0.5
    kl_qp = 0.5 * np.log(s1 / s2) + (s2 + (mu1 - mu2) ** 2) / (2 * s1) - 0.5
    # The average can dip a hair below 0 from floating-point error at dist ~ 0.
    dist = np.clip(0.5 * (kl_pq + kl_qp), 0.0, None).astype(np.float32)

    sim = _distance_block_to_similarity(dist, i_start, i_end, sim_formula)
    return i_start, i_end, sim


def _compute_block_median_iqr(
    i_start: int,
    i_end: int,
    median: np.ndarray,
    iqr: np.ndarray,
    sim_formula: int,
) -> tuple[int, int, np.ndarray]:
    """
    Robust distance for rows [i_start, i_end):
    |median_i - median_j| / (0.5 * (IQR_i + IQR_j)).
    """
    m1 = median[i_start:i_end, None]
    m2 = median[None, :]
    iqr1 = iqr[i_start:i_end, None]
    iqr2 = iqr[None, :]

    dist = (np.abs(m1 - m2) / (0.5 * (iqr1 + iqr2))).astype(np.float32)

    sim = _distance_block_to_similarity(dist, i_start, i_end, sim_formula)
    return i_start, i_end, sim


def save_parcel_order(parcel_ids: list[str], output_path: Path) -> None:
    atomic_write_text(output_path, "\n".join(parcel_ids) + "\n")


def save_metadata(
    output_path: Path,
    subject_id: str,
    n_parcels: int,
    method: str,
    sim_formula: int,
    save_matrix: bool,
    input_completion: dict,
    parcel_order_hash: str,
) -> None:
    metadata = {
        "subject_id": subject_id,
        "method": method,
        "n_parcels": n_parcels,
        "node_feature": METHOD_NODE_FEATURE_LABELS[method],
        "jacobian_type": "log-Jacobian determinant",
        "quantile_count": QUANTILE_LEN if method == METHOD_WASSERSTEIN else None,
        "distance": METHOD_LABELS[method],
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
    method: str,
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
        "method": method,
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


_METHOD_BLOCK_FN = {
    METHOD_WASSERSTEIN: _compute_block,
    METHOD_KL: _compute_block_kl,
    METHOD_MEDIAN_IQR: _compute_block_median_iqr,
}


def _block_args(method: str, i_start: int, i_end: int, sorted_matrix: np.ndarray, stats: dict, sim_formula: int) -> tuple:
    if method == METHOD_WASSERSTEIN:
        return (i_start, i_end, sorted_matrix, sim_formula)
    if method == METHOD_KL:
        return (i_start, i_end, stats["mean"], stats["var"], sim_formula)
    if method == METHOD_MEDIAN_IQR:
        return (i_start, i_end, stats["median"], stats["iqr"], sim_formula)
    raise ValueError(f"Unsupported method: {method}")


def build_one_graph(
    *,
    method: str,
    args: argparse.Namespace,
    subject_id: str,
    parcel_ids: list[str],
    sorted_matrix: np.ndarray,
    stats: dict,
    blocks: list[tuple[int, int]],
    out_dir: Path,
    input_completion: dict,
    parcel_order_hash: str,
    save_matrix: bool,
) -> None:
    N = len(parcel_ids)
    num_blocks = len(blocks)
    block_fn = delayed(_METHOD_BLOCK_FN[method])

    out_dir.mkdir(parents=True, exist_ok=True)

    mm_matrix = None
    temporary_matrix_path = out_dir / f".adjacency_matrix.{os.getpid()}.tmp"
    if save_matrix:
        mm_matrix = np.memmap(temporary_matrix_path, dtype="float32", mode="w+", shape=(N, N))
        mm_matrix[:] = 0.0
        np.fill_diagonal(mm_matrix, 1.0)

    weighted_degree = np.zeros(N, dtype=np.float64)
    start_time = time.time()

    results = Parallel(n_jobs=args.num_workers, prefer="processes", return_as="generator")(
        block_fn(*_block_args(method, i_start, i_end, sorted_matrix, stats, args.sim_formula))
        for i_start, i_end in blocks
    )

    for completed, (i_start, i_end, sim_block) in enumerate(results, start=1):
        if mm_matrix is not None:
            mm_matrix[i_start:i_end, :] = sim_block
            mm_matrix[:, i_start:i_end] = sim_block.T
        block_degree = (sim_block.sum(axis=1, dtype=np.float64) - 1.0) / (N - 1)
        if not np.isfinite(block_degree).all():
            raise ValueError(f"[{method}] Non-finite weighted degree in row block [{i_start}, {i_end})")
        weighted_degree[i_start:i_end] = block_degree

        if completed % args.progress_every == 0 or completed == num_blocks:
            elapsed = time.time() - start_time
            pct = completed / num_blocks * 100
            print(f"[{method}] [progress] {completed}/{num_blocks} blocks ({pct:.1f}%) — {elapsed:.1f}s", flush=True)

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
        method,
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
        "method": method,
        "sim_formula": args.sim_formula,
        "similarity_formula": SIM_FORMULA_LABELS[args.sim_formula],
        "adjacency_matrix_saved": save_matrix,
        "weighted_degree_dtype": "float64",
    })

    elapsed = time.time() - start_time
    print(f"[{method}] Done in {elapsed:.1f}s  →  {out_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if args.block_size < 1:
        raise ValueError("--block-size must be at least 1")
    save_matrix = args.save_matrix == "true"
    methods = parse_methods(args.method)
    if args.output_folder is not None and len(methods) != 1:
        raise ValueError("--output-folder is only valid when --method names exactly one method")

    subject_folder, subject_id = resolve_subject_folder(
        input_folder=args.input_folder,
        input_root=args.input_root,
        subject_id=args.subject_id,
    )

    print(f"Subject:      {subject_id}", flush=True)
    print(f"Input folder: {subject_folder}", flush=True)
    print(f"Methods: {', '.join(methods)}", flush=True)
    print(f"Save dense adjacency matrix: {save_matrix}", flush=True)
    print(f"Similarity formula: {args.sim_formula} ({SIM_FORMULA_LABELS[args.sim_formula]})", flush=True)

    input_completion, expected_parcel_ids = read_input_contract(
        subject_folder, args.allow_unvalidated_input
    )

    parcel_ids, sorted_matrix, stats = load_subject_parcels(subject_folder)
    N = len(parcel_ids)
    if N < 2:
        raise ValueError("At least two parcels are required to build a graph")
    parcel_order_hash = sha256_lines(parcel_ids)
    if expected_parcel_ids and parcel_ids != expected_parcel_ids:
        raise ValueError("Loaded parcel files differ from the validated input parcel order")
    print(f"Loaded {N} parcels", flush=True)

    blocks = [(i, min(i + args.block_size, N)) for i in range(0, N, args.block_size)]
    print(
        f"Computing similarities: {len(blocks)} blocks of ≤{args.block_size} rows, "
        f"{args.num_workers} workers, {len(methods)} method(s)",
        flush=True,
    )

    # Loop over all the methods the user requested, building each graph in turn. The parcel vectors are loaded once and reused.
    for method in methods:
        out_dir = args.output_folder if args.output_folder is not None else default_output_root(method, args.sim_formula, subject_id)

        # Check if the graph is already complete and valid; skip if so, unless --force is used.
        if not args.force and expected_parcel_ids and graph_is_complete(
            out_dir,
            subject_id=subject_id,
            n_parcels=N,
            parcel_order_hash=parcel_order_hash,
            input_completion=input_completion,
            method=method,
            sim_formula=args.sim_formula,
            save_matrix=save_matrix,
        ):
            print(f"[{method}] Validated graph output already complete: {out_dir}", flush=True)
            continue
        
        # Build the graph for this method, writing outputs to its designated folder.
        build_one_graph(
            method=method,
            args=args,
            subject_id=subject_id,
            parcel_ids=parcel_ids,
            sorted_matrix=sorted_matrix,
            stats=stats,
            blocks=blocks,
            out_dir=out_dir,
            input_completion=input_completion,
            parcel_order_hash=parcel_order_hash,
            save_matrix=save_matrix,
        )


if __name__ == "__main__":
    main()
