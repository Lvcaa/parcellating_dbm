"""Validate connected per-label manifests and freeze an ROI atlas contract."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_labels import KEEP_LABELS, LABEL_DICT
from pipeline_integrity import atomic_write_json, file_signature, read_json, sha256_lines


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def numeric_roi_id(path: Path) -> int:
    name = path.name
    if not name.startswith("roi_") or not name.endswith(".nii.gz"):
        raise ValueError(f"Unexpected parcel filename: {path}")
    return int(name[4:-7])


def validate_label(rois_dir: Path, label: int, parcel_size: int) -> tuple[dict, list[str]]:
    label_dir = rois_dir / str(label)
    if not label_dir.is_dir():
        raise ValueError(f"Required label folder missing: {label_dir}")
    manifest_path = label_dir / "label_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Connected-parcellation manifest missing: {manifest_path}")
    manifest = read_json(manifest_path)
    required = {
        "validation_status": "passed",
        "label": label,
        "parcel_size_target": parcel_size,
        "connectivity": 6,
        "coverage_exact": True,
        "overlap_voxels": 0,
        "parcel_sizes_match_targets": True,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Invalid {key} for label {label}: {manifest.get(key)!r}")
    records = manifest.get("parcels", [])
    files = sorted(label_dir.glob("roi_*.nii.gz"), key=numeric_roi_id)
    if len(files) != manifest.get("parcel_count") or len(files) != len(records):
        raise ValueError(f"Parcel-count mismatch for label {label}")
    record_names = [record.get("file") for record in records]
    if [path.name for path in files] != record_names:
        raise ValueError(f"Parcel files differ from label manifest for label {label}")
    if any(record.get("component_count_6") != 1 for record in records):
        raise ValueError(f"Disconnected parcel recorded for label {label}")
    if any(record.get("voxel_count") != record.get("target_voxel_count") for record in records):
        raise ValueError(f"Parcel target-size mismatch recorded for label {label}")
    actual_hashes = [sha256_file(path) for path in files]
    if actual_hashes != [record.get("sha256") for record in records]:
        raise ValueError(f"Parcel mask content differs from label manifest for label {label}")
    parcel_ids = [f"label_{label}/{path.stem.replace('.nii', '')}" for path in files]
    summary = {
        "label": label,
        "label_name": LABEL_DICT[label],
        "parcel_count": len(files),
        "source_voxel_count": manifest.get("source_voxel_count"),
        "parcel_size_min": manifest.get("parcel_size_min"),
        "parcel_size_max": manifest.get("parcel_size_max"),
        "mask_content_sha256": sha256_lines(
            f"{parcel_id}:{digest}" for parcel_id, digest in zip(parcel_ids, actual_hashes)
        ),
        "label_manifest": str(manifest_path.resolve()),
    }
    return summary, parcel_ids


def create_atlas_manifest(
    rois_dir: Path,
    segmentation: Path,
    parcel_size: int,
    output: Path,
) -> dict:
    required_labels = list(KEEP_LABELS)
    discovered_labels = sorted(
        int(path.name) for path in rois_dir.iterdir() if path.is_dir() and path.name.isdigit()
    )
    if discovered_labels != sorted(required_labels):
        missing = sorted(set(required_labels) - set(discovered_labels))
        extras = sorted(set(discovered_labels) - set(required_labels))
        raise ValueError(f"Atlas labels differ from KEEP_LABELS; missing={missing}, extras={extras}")

    label_summaries = []
    all_parcel_ids: list[str] = []
    # Preserve the exact numeric label/ROI order used by extraction and graph building.
    for label in sorted(required_labels):
        summary, parcel_ids = validate_label(rois_dir, label, parcel_size)
        label_summaries.append(summary)
        all_parcel_ids.extend(parcel_ids)

    segmentation_signature = file_signature(segmentation)
    order_hash = sha256_lines(all_parcel_ids)
    atlas_seed = f"{segmentation_signature['sha256']}:{parcel_size}:{order_hash}".encode()
    atlas_id = hashlib.sha256(atlas_seed).hexdigest()[:16]
    payload = {
        "schema_version": 1,
        "validation_status": "passed",
        "atlas_id": atlas_id,
        "rois_dir": str(rois_dir.resolve()),
        "source_segmentation": segmentation_signature,
        "required_labels": required_labels,
        "parcel_size_target": parcel_size,
        "parcel_count": len(all_parcel_ids),
        "parcel_order_sha256": order_hash,
        "connectivity": 6,
        "coverage_exact_per_label": True,
        "overlap_voxels_per_label": 0,
        "labels": label_summaries,
        "mask_content_sha256": sha256_lines(
            f"{summary['label']}:{summary['mask_content_sha256']}"
            for summary in label_summaries
        ),
    }
    atomic_write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rois-dir", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--parcel-size", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.rois_dir / "atlas_manifest.json"
    manifest = create_atlas_manifest(args.rois_dir, args.segmentation, args.parcel_size, output)
    print(f"Atlas validation passed: {manifest['atlas_id']}")
    print(f"Labels: {len(manifest['labels'])}; parcels: {manifest['parcel_count']}")
    print(f"Manifest: {output}")


if __name__ == "__main__":
    main()
