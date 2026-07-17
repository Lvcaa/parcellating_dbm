"""Manifest-aligned parcelwise inference for graph and direct DBM features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


FEATURE_LOCATIONS = {
    "weighted_degree_expW": ("wasserstein_graphs_expW", "weighted_degree.dat"),
    "weighted_degree_inv1pW": ("wasserstein_graphs_inv1pW", "weighted_degree.dat"),
    "direct_mean": ("jacobian_parcel_vectors", "direct_mean.dat"),
    "direct_median": ("jacobian_parcel_vectors", "direct_median.dat"),
}


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """Apply BH-FDR only to finite p-values and preserve invalid entries as NaN."""
    p_values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(p_values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(p_values)
    if finite.any():
        adjusted[finite] = multipletests(p_values[finite], alpha=0.05, method="fdr_bh")[1]
    return adjusted


def read_subject_manifest(path: Path) -> list[str]:
    subjects = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(subjects) != len(set(subjects)):
        raise ValueError("Subject manifest contains duplicates")
    return subjects


def cohort_table(subjects: list[str], database_path: Path) -> pd.DataFrame:
    database = pd.read_csv(database_path).copy()
    database["subject_id"] = "sub-" + database["OASISID"].astype(str).str.removeprefix("OAS3")
    if database["subject_id"].duplicated().any():
        raise ValueError("Database contains duplicate subject IDs")
    cohort = database.set_index("subject_id").loc[subjects].reset_index()
    counts = cohort["HStatus"].value_counts().to_dict()
    if counts != {"Healthy": 137, "Unhealthy": 130}:
        raise ValueError(f"Expected 137 Healthy/130 Unhealthy, got {counts}")
    return cohort


def load_feature_matrix(
    run_root: Path,
    subjects: list[str],
    feature: str,
) -> tuple[list[str], np.ndarray]:
    if feature not in FEATURE_LOCATIONS:
        raise ValueError(f"Unknown feature: {feature}")
    validation_path = run_root / "cohort_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("validation_status") != "passed":
        raise ValueError("Run has not passed cohort validation")
    n_parcels = int(validation["parcel_count"])
    folder, filename = FEATURE_LOCATIONS[feature]
    root = run_root / folder
    first_order = root / subjects[0] / "parcel_order.txt"
    if not first_order.is_file() and feature.startswith("direct_"):
        first_order = run_root / "jacobian_parcel_vectors" / subjects[0] / "parcel_order.txt"
    parcel_ids = first_order.read_text(encoding="utf-8").splitlines()
    if len(parcel_ids) != n_parcels:
        raise ValueError("Parcel-order length differs from validated parcel count")

    matrix = np.empty((len(subjects), n_parcels), dtype=np.float64)
    expected_bytes = n_parcels * np.dtype("float64").itemsize
    for row, subject in enumerate(subjects):
        path = root / subject / filename
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Feature size mismatch: {path}")
        values = np.fromfile(path, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite feature values: {path}")
        matrix[row] = values
    return parcel_ids, matrix


def welch_results(
    matrix: np.ndarray,
    cohort: pd.DataFrame,
    parcel_ids: list[str],
) -> pd.DataFrame:
    healthy = matrix[cohort["HStatus"].eq("Healthy").to_numpy()]
    unhealthy = matrix[cohort["HStatus"].eq("Unhealthy").to_numpy()]
    t_stat, p_value = stats.ttest_ind(healthy, unhealthy, axis=0, equal_var=False)
    mean_h = healthy.mean(axis=0)
    mean_u = unhealthy.mean(axis=0)
    var_h = healthy.var(axis=0, ddof=1)
    var_u = unhealthy.var(axis=0, ddof=1)
    se2_h = var_h / len(healthy)
    se2_u = var_u / len(unhealthy)
    se = np.sqrt(se2_h + se2_u)
    welch_df = (se2_h + se2_u) ** 2 / (
        se2_h**2 / (len(healthy) - 1) + se2_u**2 / (len(unhealthy) - 1)
    )
    delta = mean_u - mean_h
    critical = stats.t.ppf(0.975, welch_df)
    pooled_sd = np.sqrt(
        ((len(healthy) - 1) * var_h + (len(unhealthy) - 1) * var_u)
        / (len(healthy) + len(unhealthy) - 2)
    )
    correction = 1 - 3 / (4 * (len(healthy) + len(unhealthy)) - 9)
    hedges_g = np.divide(delta, pooled_sd, out=np.full_like(delta, np.nan), where=pooled_sd > 0)
    hedges_g *= correction
    p_fdr = fdr_bh(p_value)
    return pd.DataFrame({
        "parcel_id": parcel_ids,
        "healthy_mean": mean_h,
        "unhealthy_mean": mean_u,
        "unhealthy_minus_healthy": delta,
        "difference_ci_low": delta - critical * se,
        "difference_ci_high": delta + critical * se,
        "hedges_g": hedges_g,
        "welch_t_healthy_minus_unhealthy": t_stat,
        "welch_df": welch_df,
        "p_value": p_value,
        "p_fdr_bh": p_fdr,
        "significant_fdr_05": p_fdr < 0.05,
    })


def covariate_glm(matrix: np.ndarray, cohort: pd.DataFrame) -> pd.DataFrame:
    covariates = cohort[["HStatus", "age at visit", "GENDER", "EDUC"]].copy()
    keep = covariates.notna().all(axis=1).to_numpy()
    covariates = covariates.loc[keep]
    y = matrix[keep]
    age = covariates["age at visit"].astype(float).to_numpy()
    education = covariates["EDUC"].astype(float).to_numpy()
    age = (age - age.mean()) / age.std(ddof=1)
    education = (education - education.mean()) / education.std(ddof=1)
    x = np.column_stack([
        np.ones(len(covariates)),
        covariates["HStatus"].eq("Unhealthy").astype(float),
        age,
        covariates["GENDER"].eq("Male").astype(float),
        education,
    ])
    xtx_inverse = np.linalg.pinv(x.T @ x)
    beta = xtx_inverse @ x.T @ y
    residual = y - x @ beta
    dof = len(y) - x.shape[1]
    residual_variance = np.sum(residual**2, axis=0) / dof
    se_group = np.sqrt(residual_variance * xtx_inverse[1, 1])
    t_group = np.divide(
        beta[1], se_group, out=np.full_like(se_group, np.nan), where=se_group > 0
    )
    p_group = 2 * stats.t.sf(np.abs(t_group), dof)
    p_fdr = fdr_bh(p_group)
    return pd.DataFrame({
        "glm_beta_unhealthy": beta[1],
        "glm_t_unhealthy": t_group,
        "glm_p_value": p_group,
        "glm_p_fdr_bh": p_fdr,
        "glm_significant_fdr_05": p_fdr < 0.05,
        "glm_n": len(y),
    })


def leave_one_out_sensitivity(
    matrix: np.ndarray,
    cohort: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    candidates = np.flatnonzero(results["significant_fdr_05"].to_numpy())
    if candidates.size == 0:
        return pd.DataFrame(columns=["parcel_id", "loo_same_direction_fraction", "loo_p_lt_05_fraction"])
    signs = []
    significant = []
    labels = cohort["HStatus"].to_numpy()
    baseline_sign = np.sign(results.loc[candidates, "unhealthy_minus_healthy"].to_numpy())
    for omitted in range(len(cohort)):
        keep = np.arange(len(cohort)) != omitted
        h = matrix[keep][:, candidates][labels[keep] == "Healthy"]
        u = matrix[keep][:, candidates][labels[keep] == "Unhealthy"]
        _, p = stats.ttest_ind(h, u, axis=0, equal_var=False)
        signs.append(np.sign(u.mean(axis=0) - h.mean(axis=0)) == baseline_sign)
        significant.append(p < 0.05)
    return pd.DataFrame({
        "parcel_id": results.loc[candidates, "parcel_id"].to_numpy(),
        "loo_same_direction_fraction": np.mean(signs, axis=0),
        "loo_p_lt_05_fraction": np.mean(significant, axis=0),
    })


def analyse_feature(
    run_root: Path,
    subjects_path: Path,
    database_path: Path,
    feature: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = read_subject_manifest(subjects_path)
    cohort = cohort_table(subjects, database_path)
    parcel_ids, matrix = load_feature_matrix(run_root, subjects, feature)
    results = welch_results(matrix, cohort, parcel_ids)
    results = pd.concat([results, covariate_glm(matrix, cohort)], axis=1)
    sensitivity = leave_one_out_sensitivity(matrix, cohort, results)
    return results, sensitivity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--feature", choices=tuple(FEATURE_LOCATIONS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, sensitivity = analyse_feature(
        args.run_root, args.subjects, args.database, args.feature
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / f"{args.feature}_parcel_statistics.csv", index=False)
    sensitivity.to_csv(args.output_dir / f"{args.feature}_leave_one_out.csv", index=False)
    print(f"{args.feature}: {int(results['significant_fdr_05'].sum())} Welch/BH discoveries")
    print(f"{args.feature}: {int(results['glm_significant_fdr_05'].sum())} covariate-GLM/BH discoveries")


if __name__ == "__main__":
    main()
