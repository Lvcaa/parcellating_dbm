"""Build parcel-similarity weighted degrees for one subject.

Supported distances are approximate Wasserstein-1 (``was``), symmetrized
Gaussian KL (``kl``), and a robust median/IQR distance (``meiq``). Pairwise
similarities are computed in row blocks; only normalized weighted-degree
vectors are retained.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_integrity import (
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
            "Build one or more parcel-similarity weighted-degree vectors "
            "(Wasserstein / KL / median-IQR)."
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
        help=(
            "Root directory containing subject parcel-vector folders "
            f"(default: {DEFAULT_INPUT_ROOT})."
        ),
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
        "--force", action="store_true", help="Recompute a matching completed graph."
    )
    return parser.parse_args()


def parse_methods(raw: str) -> list[str]:
    methods: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in METHOD_CHOICES:
            raise ValueError(
                f"Unsupported --method value: {token!r}; choose from {METHOD_CHOICES}"
            )
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
    sample_count = len(v_sorted)
    if sample_count == QUANTILE_LEN:
        return v_sorted

    src_q = (np.arange(sample_count, dtype=np.float32) + 0.5) / sample_count
    tgt_q = (np.arange(QUANTILE_LEN, dtype=np.float32) + 0.5) / QUANTILE_LEN
    return np.interp(tgt_q, src_q, v_sorted).astype(np.float32)


def load_subject_parcels(
    subject_folder: Path,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """
    Return parcel IDs, a float32 sorted quantile matrix of shape (N, QUANTILE_LEN)
    for the Wasserstein method, and a dict of per-parcel summary arrays
    ("mean", "var", "median", "iqr") for the KL and median/IQR methods.
    """
    order_path = subject_folder / "parcel_order.txt"
    data_path = subject_folder / "parcel_vectors.dat"
    offsets_path = subject_folder / "parcel_offsets.dat"
    if not order_path.is_file() or not data_path.is_file() or not offsets_path.is_file():
        raise ValueError(
            f"Missing parcel_order.txt or parcel vector .dat files under {subject_folder}"
        )

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
    block = sorted_matrix[i_start:i_end]
    n_parcels, n_quantiles = sorted_matrix.shape
    dist = np.zeros((len(block), n_parcels), dtype=np.float32)
    for k in range(n_quantiles):
        dist += np.abs(block[:, k : k + 1] - sorted_matrix[:, k])
    dist /= n_quantiles

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


def read_input_contract(subject_folder: Path, parcel_ids: list[str]) -> dict:
    completion = load_completion(subject_folder)
    if completion is None:
        raise ValueError(
            f"Validated parcel completion marker missing: {subject_folder / 'complete.json'}"
        )
    if completion.get("stage") != "parcel_vectors":
        raise ValueError(f"Unexpected input completion stage in {subject_folder}")
    if len(parcel_ids) != completion.get("parcel_count"):
        raise ValueError(f"Parcel-order count differs from completion manifest: {subject_folder}")
    if sha256_lines(parcel_ids) != completion.get("parcel_order_sha256"):
        raise ValueError(f"Parcel-order hash differs from completion manifest: {subject_folder}")
    return completion


def graph_contract(
    subject_id: str,
    n_parcels: int,
    parcel_order_hash: str,
    input_completion: dict,
    method: str,
    sim_formula: int,
) -> dict:
    return {
        "schema_version": 1,
        "stage": "wasserstein_graph",
        "subject_id": subject_id,
        "parcel_count": n_parcels,
        "parcel_order_sha256": parcel_order_hash,
        "atlas_id": input_completion.get("atlas_id"),
        "atlas_manifest_sha256": input_completion.get("atlas_manifest_sha256"),
        "method": method,
        "sim_formula": sim_formula,
        "similarity_formula": SIM_FORMULA_LABELS[sim_formula],
        "weighted_degree_dtype": "float64",
    }


def graph_is_complete(out_dir: Path, contract: dict) -> bool:
    completion = load_completion(out_dir)
    if completion is None or any(
        completion.get(key) != value for key, value in contract.items()
    ):
        return False
    try:
        validate_raw_vector(
            out_dir / "weighted_degree.dat", length=contract["parcel_count"]
        )
    except ValueError:
        return False
    return not (out_dir / "adjacency_matrix.dat").exists()


def save_metadata(output_path: Path, contract: dict) -> None:
    method = contract["method"]
    metadata = {
        "subject_id": contract["subject_id"],
        "method": method,
        "n_parcels": contract["parcel_count"],
        "node_feature": METHOD_NODE_FEATURE_LABELS[method],
        "jacobian_type": "log-Jacobian determinant",
        "distance": METHOD_LABELS[method],
        "similarity_formula": contract["similarity_formula"],
        "weighted_degree_dtype": "float64",
        "weighted_degree_normalization": (
            "sum of off-diagonal similarities divided by N-1"
        ),
        "self_loops_in_weighted_degree": False,
        "atlas_id": contract["atlas_id"],
        "atlas_manifest_sha256": contract["atlas_manifest_sha256"],
        "parcel_order_sha256": contract["parcel_order_sha256"],
    }
    if method == METHOD_WASSERSTEIN:
        metadata["quantile_count"] = QUANTILE_LEN
    atomic_write_json(output_path, metadata)


def build_one_graph(
    *,
    method: str,
    args: argparse.Namespace,
    parcel_ids: list[str],
    sorted_matrix: np.ndarray,
    stats: dict,
    blocks: list[tuple[int, int]],
    out_dir: Path,
    contract: dict,
) -> None:
    n_parcels = len(parcel_ids)
    if method == METHOD_WASSERSTEIN:
        block_fn, method_args = _compute_block, (sorted_matrix,)
    elif method == METHOD_KL:
        block_fn, method_args = _compute_block_kl, (stats["mean"], stats["var"])
    else:
        block_fn = _compute_block_median_iqr
        method_args = (stats["median"], stats["iqr"])

    out_dir.mkdir(parents=True, exist_ok=True)
    weighted_degree = np.zeros(n_parcels, dtype=np.float64)
    start_time = time.time()
    results = Parallel(n_jobs=args.num_workers, prefer="processes", return_as="generator")(
        delayed(block_fn)(i_start, i_end, *method_args, args.sim_formula)
        for i_start, i_end in blocks
    )

    for completed, (i_start, i_end, similarities) in enumerate(results, start=1):
        degree = (
            similarities.sum(axis=1, dtype=np.float64) - 1.0
        ) / (n_parcels - 1)
        if not np.isfinite(degree).all():
            raise ValueError(
                f"[{method}] Non-finite weighted degree in row block "
                f"[{i_start}, {i_end})"
            )
        weighted_degree[i_start:i_end] = degree

        if completed % 10 == 0 or completed == len(blocks):
            elapsed = time.time() - start_time
            percentage = completed / len(blocks) * 100
            print(
                f"[{method}] [progress] {completed}/{len(blocks)} blocks "
                f"({percentage:.1f}%) — {elapsed:.1f}s",
                flush=True,
            )

    atomic_write_raw(out_dir / "weighted_degree.dat", weighted_degree, "float64")
    save_metadata(out_dir / "metadata.json", contract)
    atomic_write_text(out_dir / "parcel_order.txt", "\n".join(parcel_ids) + "\n")
    write_completion(out_dir, contract)
    print(f"[{method}] Done in {time.time() - start_time:.1f}s  →  {out_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if args.block_size < 1:
        raise ValueError("--block-size must be at least 1")

    methods = parse_methods(args.method)
    if args.output_folder is not None and len(methods) != 1:
        raise ValueError("--output-folder is only valid with exactly one method")

    subject_folder, subject_id = resolve_subject_folder(
        args.input_folder, args.input_root, args.subject_id
    )
    print(f"Subject:      {subject_id}", flush=True)
    print(f"Input folder: {subject_folder}", flush=True)
    print(f"Methods:      {', '.join(methods)}", flush=True)
    print(
        f"Similarity:   {SIM_FORMULA_LABELS[args.sim_formula]}",
        flush=True,
    )

    parcel_ids, sorted_matrix, stats = load_subject_parcels(subject_folder)
    input_completion = read_input_contract(subject_folder, parcel_ids)
    n_parcels = len(parcel_ids)
    if n_parcels < 2:
        raise ValueError("At least two parcels are required to build a graph")
    parcel_order_hash = sha256_lines(parcel_ids)
    blocks = [
        (start, min(start + args.block_size, n_parcels))
        for start in range(0, n_parcels, args.block_size)
    ]
    print(
        f"Loaded {n_parcels} parcels; computing {len(blocks)} blocks with "
        f"{args.num_workers} workers",
        flush=True,
    )

    for method in methods:
        out_dir = args.output_folder or default_output_root(
            method, args.sim_formula, subject_id
        )
        contract = graph_contract(
            subject_id,
            n_parcels,
            parcel_order_hash,
            input_completion,
            method,
            args.sim_formula,
        )
        if not args.force and graph_is_complete(out_dir, contract):
            print(f"[{method}] Output already complete: {out_dir}", flush=True)
            continue
        build_one_graph(
            method=method,
            args=args,
            parcel_ids=parcel_ids,
            sorted_matrix=sorted_matrix,
            stats=stats,
            blocks=blocks,
            out_dir=out_dir,
            contract=contract,
        )


if __name__ == "__main__":
    main()
