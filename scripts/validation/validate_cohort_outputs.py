"""Validate a complete balanced cohort before parcelwise inference."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_integrity import atomic_write_json, load_completion, read_json, validate_raw_vector


# Mirrors METHOD_OUTPUT_PREFIX / SIM_FORMULA_OUTPUT_NAMES in
# scripts/graph_building/wasserstein_distance_graph2.py.
GRAPH_METHOD_PREFIXES = {
    "was": "wasserstein_graphs",
    "kl": "kl_graphs",
    "meiq": "median_iqr_graphs",
}
SIM_FORMULA_NAMES = {1: "expW", 2: "inv1pW"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--expected-healthy", type=int, default=684)
    parser.add_argument("--expected-unhealthy", type=int, default=504)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Frozen run manifest to validate against (default: <run-root>/run_manifest.json).",
    )
    parser.add_argument(
        "--methods",
        default="was,kl,meiq",
        help=(
            "Comma-separated subset of {was, kl, meiq} to validate (default: was,kl,meiq, "
            "i.e. the definitive cross-definition validation). Passing a strict subset, "
            "e.g. 'was', validates a provisional single-method run instead and the report "
            "is marked non-definitive."
        ),
    )
    return parser.parse_args()


def read_subjects(path: Path) -> list[str]:
    subjects = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(subjects) != len(set(subjects)):
        raise ValueError("Subject manifest contains duplicates")
    return subjects


def read_groups(path: Path) -> dict[str, str]:
    groups = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            subject_id = "sub-" + row["OASISID"].removeprefix("OAS3")
            if subject_id in groups:
                raise ValueError(f"Duplicate database subject: {subject_id}")
            groups[subject_id] = row["HStatus"]
    return groups


def main() -> None:
    args = parse_args()
    methods = [token.strip() for token in args.methods.split(",") if token.strip()]
    unknown = [method for method in methods if method not in GRAPH_METHOD_PREFIXES]
    if not methods or unknown:
        raise ValueError(f"--methods must be a non-empty subset of {tuple(GRAPH_METHOD_PREFIXES)}")
    is_definitive = set(methods) == set(GRAPH_METHOD_PREFIXES)

    run_manifest_path = args.manifest or args.run_root / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise ValueError(f"Frozen run manifest missing: {run_manifest_path}")
    run_manifest = read_json(run_manifest_path)
    configuration = run_manifest.get("configuration", {})
    if configuration.get("sim_formula") != "both":
        raise ValueError("Cohort validation requires both similarity formulas")
    if configuration.get("save_matrix") != "false":
        raise ValueError("Cohort validation requires save_matrix=false")
    if configuration.get("method") != args.methods:
        raise ValueError(f"Run manifest method {configuration.get('method')!r} does not match --methods {args.methods!r}")
    if is_definitive:
        print("Validating all three edge definitions (was, kl, meiq): definitive cohort validation.")
    else:
        print(f"Validating methods {methods}: PROVISIONAL run, not a substitute for cross-definition validation.")

    subjects = read_subjects(args.subjects)
    groups = read_groups(args.database)
    missing_labels = [subject for subject in subjects if subject not in groups]
    if missing_labels:
        raise ValueError(f"Subjects missing database labels: {missing_labels}")
    counts = Counter(groups[subject] for subject in subjects)
    expected_counts = {"Healthy": args.expected_healthy, "Unhealthy": args.expected_unhealthy}
    if dict(counts) != expected_counts:
        raise ValueError(f"Group counts {dict(counts)} do not equal {expected_counts}")

    expected_set = set(subjects)
    vector_root = args.run_root / "jacobian_parcel_vectors"
    graph_roots = {
        f"{method}_{formula_name}": (
            args.run_root / f"{GRAPH_METHOD_PREFIXES[method]}_{formula_name}", method, formula_int
        )
        for method in methods
        for formula_int, formula_name in SIM_FORMULA_NAMES.items()
    }
    for root in (vector_root, *(entry[0] for entry in graph_roots.values())):
        discovered = {path.name for path in root.iterdir() if path.is_dir()}
        if discovered != expected_set:
            raise ValueError(
                f"Subject folders differ at {root}; missing={sorted(expected_set-discovered)}, "
                f"extras={sorted(discovered-expected_set)}"
            )

    reference_order_hash = None
    reference_atlas_hash = None
    parcel_count = None
    for subject in subjects:
        vector_completion = load_completion(vector_root / subject)
        if vector_completion is None or vector_completion.get("stage") != "parcel_vectors":
            raise ValueError(f"Incomplete parcel vectors: {subject}")
        n = int(vector_completion["parcel_count"])
        order_hash = vector_completion["parcel_order_sha256"]
        atlas_hash = vector_completion["atlas_id"]
        if parcel_count is None:
            parcel_count, reference_order_hash, reference_atlas_hash = n, order_hash, atlas_hash
        if (n, order_hash, atlas_hash) != (parcel_count, reference_order_hash, reference_atlas_hash):
            raise ValueError(f"Parcel contract differs for {subject}")
        validate_raw_vector(vector_root / subject / "direct_mean.dat", length=n)
        validate_raw_vector(vector_root / subject / "direct_median.dat", length=n)

        for label, (root, method, expected_formula) in graph_roots.items():
            completion = load_completion(root / subject)
            if completion is None or completion.get("stage") != "wasserstein_graph":
                raise ValueError(f"Incomplete {label} graph: {subject}")
            if completion.get("method") != method:
                raise ValueError(f"Wrong method in {label}/{subject}")
            if completion.get("sim_formula") != expected_formula:
                raise ValueError(f"Wrong similarity formula in {label}/{subject}")
            if completion.get("adjacency_matrix_saved") is not False:
                raise ValueError(f"Unexpected dense adjacency matrix contract: {label}/{subject}")
            if (root / subject / "adjacency_matrix.dat").exists():
                raise ValueError(f"Dense adjacency matrix retained: {label}/{subject}")
            if (
                completion.get("parcel_count"),
                completion.get("parcel_order_sha256"),
                completion.get("atlas_id"),
            ) != (parcel_count, reference_order_hash, reference_atlas_hash):
                raise ValueError(f"Graph contract differs: {label}/{subject}")
            validate_raw_vector(root / subject / "weighted_degree.dat", length=n)

    if reference_atlas_hash not in run_manifest.get("atlas_manifest", {}).values():
        raise ValueError("Subject atlas id is not present in the frozen run manifest")

    report = {
        "schema_version": 1,
        "validation_status": "passed",
        "definitive": is_definitive,
        "methods_validated": methods,
        "run_root": str(args.run_root.resolve()),
        "subject_count": len(subjects),
        "group_counts": dict(counts),
        "parcel_count": parcel_count,
        "parcel_order_sha256": reference_order_hash,
        "atlas_id": reference_atlas_hash,
        "dense_adjacency_required": False,
        "run_configuration": configuration,
    }
    report_path = args.report or args.run_root / "cohort_validation.json"
    atomic_write_json(report_path, report)
    print(f"Cohort validation passed: {len(subjects)} subjects, {parcel_count} parcels")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
