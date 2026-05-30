"""
Rename ROI parcel masks with globally unique sequential IDs.

Numeric label folders under ``outputs/rois`` are processed in order. File
contents are unchanged. A lookup CSV records the global ID range assigned to
each label. Run with ``--dry-run`` first because the default operation renames
files in place.

Usage:
    python scripts/parcellation/assign_unique_labels.py [options]

Parameters:
    --rois-dir PATH    Directory containing numeric label folders
                       (default: outputs/rois).
    --csv-path PATH    Destination lookup CSV (default: docs/label_lookup.csv).
    --dry-run          Print proposed changes without renaming or writing CSV.

Examples:
    python scripts/parcellation/assign_unique_labels.py --dry-run
    python scripts/parcellation/assign_unique_labels.py
    python scripts/parcellation/assign_unique_labels.py --rois-dir outputs/custom_rois --csv-path docs/custom_label_lookup.csv --dry-run
"""

import csv
import sys
from pathlib import Path

# Import label dictionary from the preprocessing script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preprocessing"))
from inspect_labels import label_dict  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROIS_DIR = PROJECT_ROOT / "outputs" / "rois"
DEFAULT_CSV_PATH = PROJECT_ROOT / "docs" / "label_lookup.csv"

CSV_COLUMNS = ["label_number", "label_name", "roi_id_start", "roi_id_end", "parcel_count"]

# The following functions are from the aggregate_parcels.py script,
# included here for reference
def assign_unique_labels(
    rois_dir: Path,
    csv_path: Path,
    dry_run: bool = False,
) -> None:

    # Get a list of all parcel files in the directory
    rois_dir = rois_dir.resolve()
    if not rois_dir.is_dir():
        print(f"ERROR: {rois_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Sort label folders numerically (3, 4, 5, 10, 11, ...)
    label_dirs = sorted(

        # Iterate over subdirectories in rois_dir and filter 
        # to only include directories
        (d for d in rois_dir.iterdir() if d.is_dir()),
        key=lambda d: int(d.name),
    )

    if not label_dirs:
        print("No label folders found.")
        return

    # Keep a running global ID counter to assign unique IDs across all labels
    global_id = 0

    # Prepare a list to hold CSV rows, which will be written 
    # at the end (or during the run)
    csv_rows: list[dict] = []

    # Process each label folder in sorted order
    for label_dir in label_dirs:
        label_number = int(label_dir.name)
        label_name = label_dict.get(label_number, f"Unknown ({label_number})")

        # Get a sorted list of parcel files in the current label folder
        parcels = sorted(
            label_dir.glob("roi_*.nii.gz"),
            key=lambda p: int(p.name[4:-7]),  # extract numeric ID between "roi_" and ".nii.gz"
        )
        if not parcels:
            print(f"  [{label_dir.name}] no roi_*.nii.gz files, skipping")
            continue

        print(f"[{label_number}] {label_name} — {len(parcels)} parcels")

        roi_id_start = global_id

        for parcel in parcels:
            new_name = f"roi_{global_id:04d}.nii.gz"
            new_path = label_dir / new_name

            if parcel.name == new_name:
                print(f"  {parcel.name} -> (already correct, skipping)")
                global_id += 1
                continue

            if new_path.exists():
                print(
                    f"  ERROR: target {new_name} already exists in {label_dir.name}, aborting",
                    file=sys.stderr,
                )
                sys.exit(1)

            print(f"  {parcel.name} -> {new_name}")
            if not dry_run:
                parcel.rename(new_path)

            global_id += 1

        roi_id_end = global_id - 1

        csv_rows.append({
            "label_number": label_number,
            "label_name": label_name,
            "roi_id_start": f"{roi_id_start:04d}",
            "roi_id_end": f"{roi_id_end:04d}",
            "parcel_count": len(parcels),
        })

        # Write/update the CSV after every label so it stays current
        # even if the script is interrupted mid-run.
        if not dry_run:
            _write_csv(csv_path, csv_rows)

    print(f"\nDone. Assigned {global_id} unique IDs across {len(csv_rows)} labels.")
    if dry_run:
        print("(dry-run mode — no files renamed, no CSV written)")
        _print_csv_preview(csv_rows)
    else:
        print(f"Lookup table written to {csv_path}")


def _write_csv(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _print_csv_preview(rows: list[dict]) -> None:
    print("\nCSV preview (not written in dry-run):")
    header = ",".join(CSV_COLUMNS)
    print(header)
    for row in rows:
        print(",".join(str(row[c]) for c in CSV_COLUMNS))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rois-dir",
        type=Path,
        default=DEFAULT_ROIS_DIR,
        help="Path to the rois directory (default: outputs/rois relative to repo root)",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path for the output lookup CSV (default: docs/label_lookup.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without renaming files or writing the CSV",
    )
    args = parser.parse_args()

    assign_unique_labels(args.rois_dir, args.csv_path, dry_run=args.dry_run)
