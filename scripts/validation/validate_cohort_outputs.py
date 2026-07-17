"""Validate a complete balanced cohort before parcelwise inference."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_integrity import atomic_write_json, load_completion, read_json, validate_raw_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--expected-healthy", type=int, default=137)
    parser.add_argument("--expected-unhealthy", type=int, default=130)
    parser.add_argument("--report", type=Path, default=None)
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
    run_manifest_path = args.run_root / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise ValueError(f"Frozen run manifest missing: {run_manifest_path}")
    run_manifest = read_json(run_manifest_path)
    configuration = run_manifest.get("configuration", {})
    if configuration.get("sim_formula") != "both":
        raise ValueError("Definitive cohort validation requires both similarity formulas")
    if configuration.get("save_matrix") != "false":
        raise ValueError("Definitive cohort validation requires save_matrix=false")

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
        "expW": (args.run_root / "wasserstein_graphs_expW", 1),
        "inv1pW": (args.run_root / "wasserstein_graphs_inv1pW", 2),
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
        atlas_hash = vector_completion["atlas_manifest_sha256"]
        if parcel_count is None:
            parcel_count, reference_order_hash, reference_atlas_hash = n, order_hash, atlas_hash
        if (n, order_hash, atlas_hash) != (parcel_count, reference_order_hash, reference_atlas_hash):
            raise ValueError(f"Parcel contract differs for {subject}")
        validate_raw_vector(vector_root / subject / "direct_mean.dat", length=n)
        validate_raw_vector(vector_root / subject / "direct_median.dat", length=n)

        for formula, (root, expected_formula) in graph_roots.items():
            completion = load_completion(root / subject)
            if completion is None or completion.get("stage") != "wasserstein_graph":
                raise ValueError(f"Incomplete {formula} graph: {subject}")
            if completion.get("sim_formula") != expected_formula:
                raise ValueError(f"Wrong similarity formula in {formula}/{subject}")
            if completion.get("adjacency_matrix_saved") is not False:
                raise ValueError(f"Unexpected dense adjacency matrix contract: {formula}/{subject}")
            if (root / subject / "adjacency_matrix.dat").exists():
                raise ValueError(f"Dense adjacency matrix retained: {formula}/{subject}")
            if (
                completion.get("parcel_count"),
                completion.get("parcel_order_sha256"),
                completion.get("atlas_manifest_sha256"),
            ) != (parcel_count, reference_order_hash, reference_atlas_hash):
                raise ValueError(f"Graph contract differs: {formula}/{subject}")
            validate_raw_vector(root / subject / "weighted_degree.dat", length=n)

    if reference_atlas_hash not in run_manifest.get("files", {}).values():
        raise ValueError("Subject atlas hash is not present in the frozen run manifest")

    report = {
        "schema_version": 1,
        "validation_status": "passed",
        "run_root": str(args.run_root.resolve()),
        "subject_count": len(subjects),
        "group_counts": dict(counts),
        "parcel_count": parcel_count,
        "parcel_order_sha256": reference_order_hash,
        "atlas_manifest_sha256": reference_atlas_hash,
        "dense_adjacency_required": False,
        "run_configuration": configuration,
    }
    report_path = args.report or args.run_root / "cohort_validation.json"
    atomic_write_json(report_path, report)
    print(f"Cohort validation passed: {len(subjects)} subjects, {parcel_count} parcels")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
