"""Fail fast unless a subject manifest is complete, balanced, and backed by warps."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--warps-root", type=Path, required=True)
    parser.add_argument("--expected-healthy", type=int, default=137)
    parser.add_argument("--expected-unhealthy", type=int, default=130)
    return parser.parse_args()


def read_subjects(path: Path) -> list[str]:
    subjects = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(subjects) != len(set(subjects)):
        raise ValueError("Subject manifest contains duplicate IDs")
    invalid = [subject for subject in subjects if not subject.startswith("sub-")]
    if invalid:
        raise ValueError(f"Invalid subject IDs: {invalid}")
    return subjects


def read_groups(path: Path) -> dict[str, str]:
    groups: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            subject = "sub-" + row["OASISID"].removeprefix("OAS3")
            if subject in groups:
                raise ValueError(f"Duplicate database subject: {subject}")
            groups[subject] = row["HStatus"]
    return groups


def main() -> None:
    args = parse_args()
    subjects = read_subjects(args.subjects)
    expected_total = args.expected_healthy + args.expected_unhealthy
    if len(subjects) != expected_total:
        raise ValueError(f"Expected {expected_total} subjects, found {len(subjects)}")

    groups = read_groups(args.database)
    missing_labels = [subject for subject in subjects if subject not in groups]
    if missing_labels:
        raise ValueError(f"Subjects missing database labels: {missing_labels}")
    counts = Counter(groups[subject] for subject in subjects)
    expected = {"Healthy": args.expected_healthy, "Unhealthy": args.expected_unhealthy}
    if dict(counts) != expected:
        raise ValueError(f"Expected group counts {expected}, found {dict(counts)}")

    missing_warps = [
        subject
        for subject in subjects
        if not (args.warps_root / subject / "Reg_" / "_SyN1Warp.nii.gz").is_file()
    ]
    if missing_warps:
        raise ValueError(f"Missing source warps for {len(missing_warps)} subjects: {missing_warps}")
    print(
        f"Batch preflight passed: {len(subjects)} subjects "
        f"({counts['Healthy']} Healthy/{counts['Unhealthy']} Unhealthy), all warps present"
    )


if __name__ == "__main__":
    main()
