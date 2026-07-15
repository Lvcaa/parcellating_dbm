"""
Build and persist train/test splits for healthy and unhealthy subjects.

Splits are saved as a single JSON file at the path defined by const.SPLITS_PATH:
    {
        "train_healthy":   ["0001", "0007", ...],
        "test_healthy":    ["0003", ...],
        "train_unhealthy": [...],
        "test_unhealthy":  [...]
    }

IDs are 4-digit zero-padded strings (e.g. "0001"), matching the sub-XXXX
folder names in DATASET_DIR.

Usage:
    python scripts/data_handle.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import sklearn.model_selection

sys.path.insert(0, str(Path(__file__).resolve().parent))
import const


def load_subjects() -> dict[str, dict]:
    """Return metadata for every subject in DATABASE_PATH, keyed by 4-digit ID."""
    subjects: dict[str, dict] = {}
    skipped = []
    with open(const.DATABASE_PATH, newline="") as f:
        for row in csv.DictReader(f):
            sid = row["ID"][4:]   # "OAS30001" -> "0001"
            if not row["Age"].strip():
                skipped.append(sid)
                continue
            subjects[sid] = {
                "id":      sid,
                "age":     float(row["Age"]),
                "sex":     row["Sex"],
                "healthy": row["HStatus"] == "Healthy",
            }
    if skipped:
        print(f"Skipped {len(skipped)} subjects with missing age: {skipped}")
    return subjects


def _balance_unhealthy(
    healthy_test_ids: list[str],
    subjects: dict[str, dict],
) -> tuple[list[str], list[str]]:
    """
    Match each healthy test subject to the best available unhealthy subject.
    Preference: same sex first, then closest age. Remaining unhealthy subjects
    go to the training set.
    """
    all_unhealthy = [sid for sid, s in subjects.items() if not s["healthy"]]
    available = set(all_unhealthy)
    test_ids: list[str] = []

    for hid in healthy_test_ids:
        h = subjects[hid]
        same_sex = [uid for uid in available if subjects[uid]["sex"] == h["sex"]]
        candidates = same_sex if same_sex else list(available)
        if not candidates:
            break
        best = min(candidates, key=lambda uid: abs(subjects[uid]["age"] - h["age"]))
        test_ids.append(best)
        available.remove(best)

    used = set(test_ids)
    train_ids = [uid for uid in all_unhealthy if uid not in used]
    return train_ids, test_ids


def create_splits(test_size: float = 0.2) -> dict[str, list[str]]:
    """
    Split subjects into balanced train/test sets and save to SPLITS_PATH.
    Returns the splits dict.
    """
    subjects = load_subjects()

    healthy_ids = [sid for sid, s in subjects.items() if s["healthy"]]

    train_healthy, test_healthy = sklearn.model_selection.train_test_split(
        healthy_ids,
        test_size=test_size,
        random_state=const.RANDOM_SEED,
    )

    train_unhealthy, test_unhealthy = _balance_unhealthy(test_healthy, subjects)

    splits = {
        "train_healthy":   sorted(train_healthy),
        "test_healthy":    sorted(test_healthy),
        "train_unhealthy": sorted(train_unhealthy),
        "test_unhealthy":  sorted(test_unhealthy),
    }

    const.SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(const.SPLITS_PATH, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"train_healthy:   {len(train_healthy)}")
    print(f"test_healthy:    {len(test_healthy)}")
    print(f"train_unhealthy: {len(train_unhealthy)}")
    print(f"test_unhealthy:  {len(test_unhealthy)}")
    print(f"Splits saved to {const.SPLITS_PATH}")

    return splits


def load_splits() -> dict[str, list[str]]:
    """Load splits from SPLITS_PATH."""
    with open(const.SPLITS_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    create_splits()
