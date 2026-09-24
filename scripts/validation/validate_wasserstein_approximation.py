#!/usr/bin/env python3
"""Test whether the 15-point W1 approximation changes parcel group-effect maps.

For every subject and every parcel, estimate the correction from the stored
15-point degree to the exact empirical one-dimensional Wasserstein-1 degree:

    exact degree ~= stored degree + mean_j(exp(-W1_exact) - exp(-W1_grid15))

Comparator parcels are sampled uniformly from the full atlas. Independent
replicates quantify Monte Carlo uncertainty. Exact W1 is evaluated on the GPU
without scipy's per-edge Python calls by grouping parcel pairs by voxel count
and expressing the empirical quantile integral as a weighted L1 distance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "graph_building"))
from wasserstein_distance_graph2 import QUANTILE_LEN, _to_quantile_grid


DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "cohorts"
    / "oasis3"
    / "runs"
    / "connected-atlas-8cf3dc8f"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "data" / "splits.json")
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "database_finale_labels_corrette.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT / "statistics" / "wasserstein_exact_validation",
    )
    parser.add_argument("--samples-per-replicate", type=int, default=512)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional pilot limit; subjects remain balanced between groups.",
    )
    return parser.parse_args()


def empirical_w1_plan(n: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact quantile-step indices and integration widths for sizes n,m."""
    breaks = np.unique(
        np.concatenate(
            (
                np.arange(n + 1, dtype=np.float64) / n,
                np.arange(m + 1, dtype=np.float64) / m,
            )
        )
    )
    widths = np.diff(breaks)
    midpoints = (breaks[:-1] + breaks[1:]) * 0.5
    left = np.minimum((midpoints * n).astype(np.int64), n - 1)
    right = np.minimum((midpoints * m).astype(np.int64), m - 1)
    return left, right, widths


def self_check_exact_w1() -> None:
    """Check the grouped weighted-L1 representation against scipy."""
    rng = np.random.default_rng(1847)
    for n in range(1, 21):
        for m in range(1, 21):
            left, right, widths = empirical_w1_plan(n, m)
            for _ in range(3):
                a = np.sort(rng.normal(size=n))
                b = np.sort(rng.normal(size=m))
                observed = np.sum(widths * np.abs(a[left] - b[right]))
                expected = stats.wasserstein_distance(a, b)
                if not np.isclose(observed, expected, rtol=1e-12, atol=1e-12):
                    raise AssertionError((n, m, observed, expected))


def select_cohort(
    graph_root: Path, splits_path: Path, database_path: Path, max_subjects: int | None
) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    available = {
        folder.name
        for folder in graph_root.glob("sub-*")
        if (folder / "weighted_degree.dat").is_file()
        and (folder / "parcel_order.txt").is_file()
    }
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    healthy = sorted({f"sub-{item}" for item in splits["test_healthy"]} & available)
    unhealthy = sorted({f"sub-{item}" for item in splits["test_unhealthy"]} & available)
    if max_subjects is not None:
        per_group = min(max_subjects // 2, len(healthy), len(unhealthy))
        healthy = healthy[:per_group]
        unhealthy = unhealthy[:per_group]
    subjects = healthy + unhealthy
    labels = np.array([0] * len(healthy) + [1] * len(unhealthy), dtype=np.int8)

    database = pd.read_csv(database_path)
    database["subject_id"] = "sub-" + database["OASISID"].astype(str).str.removeprefix("OAS3")
    cohort = database.set_index("subject_id").loc[subjects].reset_index()
    expected = np.where(labels == 0, "Healthy", "Unhealthy")
    if not np.array_equal(cohort["HStatus"].to_numpy(), expected):
        raise ValueError("Cohort labels do not agree with splits.json")
    return subjects, labels, cohort


def load_sorted_parcels(subject_folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.fromfile(subject_folder / "parcel_offsets.dat", dtype=np.int64)
    data = np.memmap(subject_folder / "parcel_vectors.dat", dtype=np.float32, mode="r")
    sizes = np.diff(offsets)
    maximum = int(sizes.max())
    sorted_values = np.zeros((len(sizes), maximum), dtype=np.float32)
    grid15 = np.empty((len(sizes), QUANTILE_LEN), dtype=np.float32)
    for parcel, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        values = np.sort(np.asarray(data[start:end], dtype=np.float32))
        sorted_values[parcel, : len(values)] = values
        grid15[parcel] = _to_quantile_grid(values)
    return sizes, sorted_values, grid15


def fixed_atlas_sizes(parcel_root: Path, subject: str) -> np.ndarray:
    offsets = np.fromfile(parcel_root / subject / "parcel_offsets.dat", dtype=np.int64)
    return np.diff(offsets)


def draw_comparators(
    n_parcels: int, replicates: int, samples_per_replicate: int, seed: int
) -> np.ndarray:
    if samples_per_replicate > n_parcels:
        raise ValueError("samples-per-replicate cannot exceed the parcel count")
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.choice(n_parcels, samples_per_replicate, replace=False) for _ in range(replicates)]
    )


@torch.inference_mode()
def estimate_degree_corrections(
    sorted_values_np: np.ndarray,
    grid15_np: np.ndarray,
    sizes: np.ndarray,
    comparator_indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return replicate x parcel estimates of exact-minus-grid15 degree."""
    n_parcels = len(sizes)
    replicates, samples_per_replicate = comparator_indices.shape
    flat_comparators = comparator_indices.reshape(-1)
    total_samples = len(flat_comparators)

    values = torch.from_numpy(sorted_values_np).to(device)
    grid15 = torch.from_numpy(grid15_np).to(device)
    comparator_grid = grid15[torch.as_tensor(flat_comparators, device=device)]
    correction_sums = torch.zeros((replicates, n_parcels), dtype=torch.float64, device="cpu")

    unique_sizes = np.unique(sizes)
    comparator_sizes = sizes[flat_comparators]
    plans: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for n in unique_sizes:
        for m in np.unique(comparator_sizes):
            left, right, widths = empirical_w1_plan(int(n), int(m))
            plans[(int(n), int(m))] = (
                torch.as_tensor(left, dtype=torch.long, device=device),
                torch.as_tensor(right, dtype=torch.long, device=device),
                torch.as_tensor(widths, dtype=torch.float32, device=device),
            )

    for n in unique_sizes:
        target_indices = np.flatnonzero(sizes == n)
        for start in range(0, len(target_indices), batch_size):
            batch_indices = target_indices[start : start + batch_size]
            target_t = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
            approximate_distance = torch.cdist(grid15[target_t], comparator_grid, p=1) / QUANTILE_LEN
            approximate_similarity = torch.exp(-approximate_distance)
            exact_similarity = torch.empty_like(approximate_similarity)

            for m in np.unique(comparator_sizes):
                positions = np.flatnonzero(comparator_sizes == m)
                comparator_t = torch.as_tensor(
                    flat_comparators[positions], dtype=torch.long, device=device
                )
                left, right, widths = plans[(int(n), int(m))]
                left_features = values[target_t][:, left] * widths
                right_features = values[comparator_t][:, right] * widths
                exact_distance = torch.cdist(left_features, right_features, p=1)
                exact_similarity[:, torch.as_tensor(positions, device=device)] = torch.exp(
                    -exact_distance
                )

            delta = (exact_similarity - approximate_similarity).reshape(
                len(batch_indices), replicates, samples_per_replicate
            )
            # The sampled population includes the self-edge with delta=0. Convert
            # its N-item mean to the graph's off-diagonal (N-1) normalization.
            batch_correction = delta.mean(dim=2, dtype=torch.float64) * (
                n_parcels / (n_parcels - 1)
            )
            correction_sums[:, batch_indices] = batch_correction.T.cpu()

    return correction_sums.numpy()


def fdr(p_values: np.ndarray) -> np.ndarray:
    return multipletests(p_values, method="fdr_bh")[1]


def welch_map(matrix: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    healthy = matrix[labels == 0]
    unhealthy = matrix[labels == 1]
    t_stat, p_value = stats.ttest_ind(healthy, unhealthy, axis=0, equal_var=False)
    mean_h = healthy.mean(axis=0)
    mean_u = unhealthy.mean(axis=0)
    var_h = healthy.var(axis=0, ddof=1)
    var_u = unhealthy.var(axis=0, ddof=1)
    pooled = np.sqrt(
        ((len(healthy) - 1) * var_h + (len(unhealthy) - 1) * var_u)
        / (len(healthy) + len(unhealthy) - 2)
    )
    correction = 1 - 3 / (4 * (len(healthy) + len(unhealthy)) - 9)
    effect = np.divide(mean_u - mean_h, pooled, out=np.full_like(pooled, np.nan), where=pooled > 0)
    effect *= correction
    return {
        "effect": effect,
        "p": p_value,
        "q": fdr(p_value),
        "significant": fdr(p_value) < 0.05,
        "t": t_stat,
    }


def glm_map(matrix: np.ndarray, cohort: pd.DataFrame) -> dict[str, np.ndarray]:
    covariates = cohort[["HStatus", "age at visit", "GENDER", "EDUC"]].copy()
    keep = covariates.notna().all(axis=1).to_numpy()
    covariates = covariates.loc[keep]
    y = matrix[keep]
    age = covariates["age at visit"].astype(float).to_numpy()
    education = covariates["EDUC"].astype(float).to_numpy()
    age = (age - age.mean()) / age.std(ddof=1)
    education = (education - education.mean()) / education.std(ddof=1)
    design = np.column_stack(
        (
            np.ones(len(covariates)),
            covariates["HStatus"].eq("Unhealthy").astype(float),
            age,
            covariates["GENDER"].eq("Male").astype(float),
            education,
        )
    )
    inverse = np.linalg.pinv(design.T @ design)
    beta = inverse @ design.T @ y
    residual = y - design @ beta
    dof = len(y) - design.shape[1]
    variance = np.sum(residual**2, axis=0) / dof
    se = np.sqrt(variance * inverse[1, 1])
    t_stat = np.divide(beta[1], se, out=np.full_like(se, np.nan), where=se > 0)
    p_value = 2 * stats.t.sf(np.abs(t_stat), dof)
    return {
        "effect": beta[1],
        "p": p_value,
        "q": fdr(p_value),
        "significant": fdr(p_value) < 0.05,
        "t": t_stat,
    }


def compare_maps(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict:
    ref_sig = reference["significant"]
    can_sig = candidate["significant"]
    effect_difference = candidate["effect"] - reference["effect"]
    ref_count = int(ref_sig.sum())
    overlap = int((ref_sig & can_sig).sum())
    signs = np.sign(reference["effect"]) == np.sign(candidate["effect"])
    return {
        "reference_discoveries": ref_count,
        "candidate_discoveries": int(can_sig.sum()),
        "discovery_overlap": overlap,
        "reference_discovery_retention": overlap / ref_count if ref_count else None,
        "lost_discoveries": int((ref_sig & ~can_sig).sum()),
        "new_discoveries": int((~ref_sig & can_sig).sum()),
        "effect_pearson_r": float(stats.pearsonr(reference["effect"], candidate["effect"]).statistic),
        "effect_spearman_rho": float(stats.spearmanr(reference["effect"], candidate["effect"]).statistic),
        "effect_difference_median_abs": float(np.median(np.abs(effect_difference))),
        "effect_difference_p95_abs": float(np.quantile(np.abs(effect_difference), 0.95)),
        "effect_difference_max_abs": float(np.max(np.abs(effect_difference))),
        "direction_agreement_all": float(signs.mean()),
        "direction_agreement_reference_discoveries": float(signs[ref_sig].mean()) if ref_count else None,
    }


def main() -> None:
    args = parse_args()
    if args.replicates < 2:
        raise ValueError("Use at least two replicates to quantify sampling uncertainty")
    if args.samples_per_replicate < 1 or args.batch_size < 1:
        raise ValueError("Sample and batch sizes must be positive")
    self_check_exact_w1()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    parcel_root = args.run_root / "parcel_vectors"
    graph_root = args.run_root / "graphs" / "unthresholded" / "was" / "expW"
    subjects, labels, cohort = select_cohort(
        graph_root, args.splits, args.database, args.max_subjects
    )
    first_order = (graph_root / subjects[0] / "parcel_order.txt").read_text().splitlines()
    n_parcels = len(first_order)
    atlas_sizes = fixed_atlas_sizes(parcel_root, subjects[0])
    if len(atlas_sizes) != n_parcels:
        raise ValueError("Parcel offsets and graph parcel order have different lengths")
    comparators = draw_comparators(
        n_parcels, args.replicates, args.samples_per_replicate, args.seed
    )

    approximate = np.empty((len(subjects), n_parcels), dtype=np.float64)
    corrections = np.empty((args.replicates, len(subjects), n_parcels), dtype=np.float32)
    started = time.time()
    for row, subject in enumerate(subjects):
        sizes, sorted_values, grid15 = load_sorted_parcels(parcel_root / subject)
        if not np.array_equal(sizes, atlas_sizes):
            raise ValueError(f"Parcel sizes differ from the fixed atlas: {subject}")
        approximate[row] = np.fromfile(
            graph_root / subject / "weighted_degree.dat", dtype=np.float64
        )
        corrections[:, row] = estimate_degree_corrections(
            sorted_values,
            grid15,
            sizes,
            comparators,
            device=device,
            batch_size=args.batch_size,
        ).astype(np.float32)
        elapsed = time.time() - started
        print(
            f"{row + 1}/{len(subjects)} {subject}: {elapsed:.1f}s total, "
            f"{elapsed / (row + 1):.2f}s/subject",
            flush=True,
        )

    corrected_replicates = approximate[None, :, :] + corrections.astype(np.float64)
    corrected = corrected_replicates.mean(axis=0)
    approx_welch = welch_map(approximate, labels)
    corrected_welch = welch_map(corrected, labels)
    approx_glm = glm_map(approximate, cohort)
    corrected_glm = glm_map(corrected, cohort)

    replicate_welch = [welch_map(item, labels) for item in corrected_replicates]
    replicate_glm = [glm_map(item, cohort) for item in corrected_replicates]
    summary = {
        "method": "uniform Monte Carlo estimate of exact empirical W1 degree correction",
        "run_root": str(args.run_root),
        "healthy_subjects": int((labels == 0).sum()),
        "unhealthy_subjects": int((labels == 1).sum()),
        "parcels": n_parcels,
        "samples_per_replicate": args.samples_per_replicate,
        "replicates": args.replicates,
        "seed": args.seed,
        "device": str(device),
        "elapsed_seconds": time.time() - started,
        "degree_correction": {
            "median_abs": float(np.median(np.abs(corrections))),
            "p95_abs": float(np.quantile(np.abs(corrections), 0.95)),
            "max_abs": float(np.max(np.abs(corrections))),
            "replicate_difference_median_abs": float(
                np.median(np.abs(corrections[0] - corrections[1]))
            ),
            "replicate_difference_p95_abs": float(
                np.quantile(np.abs(corrections[0] - corrections[1]), 0.95)
            ),
            "replicate_difference_max_abs": float(
                np.max(np.abs(corrections[0] - corrections[1]))
            ),
        },
        "welch_approx_vs_corrected": compare_maps(approx_welch, corrected_welch),
        "glm_approx_vs_corrected": compare_maps(approx_glm, corrected_glm),
        "welch_replicate_1_vs_2": compare_maps(replicate_welch[0], replicate_welch[1]),
        "glm_replicate_1_vs_2": compare_maps(replicate_glm[0], replicate_glm[1]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "degree_corrections_by_replicate.npy", corrections)
    np.save(args.output_dir / "corrected_degree_estimate.npy", corrected.astype(np.float32))
    np.save(args.output_dir / "comparator_indices.npy", comparators)
    comparison = pd.DataFrame(
        {
            "parcel_id": first_order,
            "parcel_voxels": atlas_sizes,
            "approx_welch_g": approx_welch["effect"],
            "corrected_welch_g": corrected_welch["effect"],
            "approx_welch_q": approx_welch["q"],
            "corrected_welch_q": corrected_welch["q"],
            "approx_welch_fdr": approx_welch["significant"],
            "corrected_welch_fdr": corrected_welch["significant"],
            "approx_glm_beta": approx_glm["effect"],
            "corrected_glm_beta": corrected_glm["effect"],
            "approx_glm_q": approx_glm["q"],
            "corrected_glm_q": corrected_glm["q"],
            "approx_glm_fdr": approx_glm["significant"],
            "corrected_glm_fdr": corrected_glm["significant"],
        }
    )
    comparison.to_csv(args.output_dir / "parcel_map_comparison.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
