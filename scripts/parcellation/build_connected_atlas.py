"""Build the complete bilateral 6-connected ROI atlas on a compute cluster."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_labels import KEEP_LABELS, LABEL_DICT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARCELLATOR = Path(__file__).resolve().parent / "separate_cases" / "second_roi_test.py"
VALIDATOR = PROJECT_ROOT / "scripts" / "validation" / "validate_roi_atlas.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parcel-size", type=int, default=15)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip labels that already have a passed label_manifest.json.",
    )
    return parser.parse_args()


def label_complete(output_root: Path, label: int) -> bool:
    manifest = output_root / str(label) / "label_manifest.json"
    if not manifest.is_file():
        return False
    import json

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    records = payload.get("parcels", [])
    return (
        payload.get("validation_status") == "passed"
        and payload.get("label") == label
        and payload.get("connectivity") == 6
        and payload.get("coverage_exact") is True
        and records
        and all(record.get("sha256") for record in records)
    )


def main() -> None:
    args = parse_args()
    if args.parcel_size < 1:
        raise ValueError("--parcel-size must be at least 1")
    if not args.segmentation.is_file():
        raise FileNotFoundError(args.segmentation)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for index, label in enumerate(KEEP_LABELS, start=1):
        if args.resume and label_complete(args.output_root, label):
            print(f"[{index}/{len(KEEP_LABELS)}] label {label}: validated, skipping", flush=True)
            continue
        print(
            f"[{index}/{len(KEEP_LABELS)}] label {label} ({LABEL_DICT[label]})",
            flush=True,
        )
        subprocess.run([
            sys.executable,
            str(PARCELLATOR),
            "--roi-label", str(label),
            "--parcel-size", str(args.parcel_size),
            "--segmentation", str(args.segmentation),
            "--output-root", str(args.output_root),
        ], check=True)

    subprocess.run([
        sys.executable,
        str(VALIDATOR),
        "--rois-dir", str(args.output_root),
        "--segmentation", str(args.segmentation),
        "--parcel-size", str(args.parcel_size),
    ], check=True)
    print(f"Connected bilateral atlas complete: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
