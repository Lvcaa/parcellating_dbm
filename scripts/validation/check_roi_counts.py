"""
Check that outputs/rois has every expected label and the expected parcel count,
and write that result as outputs/rois/atlas_manifest.json.

Super simple sanity check, not a cryptographic atlas freeze: for every label
in KEEP_LABELS, compare the number of roi_*.nii.gz files on disk against
ceil(voxel_count / parcel_size), computed directly from the segmentation.
The written manifest is the atlas contract export_masked_jacobian_vectors.py
requires before it will run: it refuses unless every label's status is "ok".

Usage:
    python scripts/validation/check_roi_counts.py
    python scripts/validation/check_roi_counts.py --rois-dir outputs/rois --parcel-size 15
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_labels import KEEP_LABELS, LABEL_DICT
from pipeline_integrity import atomic_write_json, sha256_lines


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTATION = PROJECT_ROOT / "data" / "reference" / "MNI152_T1_1mm_seg.nii.gz"
DEFAULT_ROIS_DIR = PROJECT_ROOT / "outputs" / "rois"
MANIFEST_FILENAME = "atlas_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", type=Path, default=DEFAULT_SEGMENTATION)
    parser.add_argument("--rois-dir", type=Path, default=DEFAULT_ROIS_DIR)
    parser.add_argument("--parcel-size", type=int, default=15)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help=f"Where to write the manifest (default: <rois-dir>/{MANIFEST_FILENAME}).",
    )
    return parser.parse_args()


def expected_parcel_counts(segmentation: Path, parcel_size: int) -> dict[int, int]:
    data = nib.load(str(segmentation)).get_fdata()
    labels, counts = np.unique(data, return_counts=True)
    voxel_counts = dict(zip(labels.astype(int), counts.astype(int)))
    return {
        label: math.ceil(voxel_counts.get(label, 0) / parcel_size)
        for label in KEEP_LABELS
    }


def actual_parcel_count(rois_dir: Path, label: int) -> int | None:
    label_dir = rois_dir / str(label)
    if not label_dir.is_dir():
        return None
    return len(list(label_dir.glob("roi_*.nii.gz")))


def main() -> None:
    args = parse_args()
    if args.parcel_size < 1:
        raise ValueError("--parcel-size must be at least 1")

    expected = expected_parcel_counts(args.segmentation, args.parcel_size)

    print(f"{'label':>6} {'name':<32} {'expected':>9} {'actual':>9}  status")
    rows = []
    all_ok = True
    for label in KEEP_LABELS:
        actual = actual_parcel_count(args.rois_dir, label)
        exp = expected[label]
        if actual is None:
            status, all_ok = "MISSING", False
        elif actual != exp:
            status, all_ok = "MISMATCH", False
        else:
            status = "ok"
        print(f"{label:>6} {LABEL_DICT[label]:<32} {exp:>9} {str(actual):>9}  {status}")
        rows.append({
            "label": label,
            "name": LABEL_DICT[label],
            "expected": exp,
            "actual": actual,
            "status": status,
        })

    atlas_id = sha256_lines(
        f"{row['label']}:{row['expected']}:{row['actual']}" for row in rows
    )[:16]
    now = datetime.now()
    manifest_path = args.manifest_output or args.rois_dir / MANIFEST_FILENAME
    atomic_write_json(manifest_path, {
        "schema_version": 1,
        "atlas_id": atlas_id,
        "run_date": now.strftime("%d-%m-%Y"),
        "run_time": now.strftime("%H:%M"),
        "segmentation": str(args.segmentation.resolve()),
        "rois_dir": str(args.rois_dir.resolve()),
        "parcel_size": args.parcel_size,
        "all_ok": all_ok,
        "labels": rows,
    })
    print(f"\nManifest written: {manifest_path}")

    if not all_ok:
        print(f"FAILED: {args.rois_dir} is incomplete or inconsistent with the segmentation.")
        sys.exit(1)
    print(f"PASSED: every label in {args.rois_dir} has the expected parcel count.")


if __name__ == "__main__":
    main()
