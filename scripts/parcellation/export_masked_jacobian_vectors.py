"""
Extract parcel-level Jacobian vectors into one concatenated memmap per subject.

The script supports a single Jacobian image or batch processing of matching
images in a directory. Outputs are written under ``<output-dir>/<subject-id>/``
as ``parcel_vectors.dat`` (float32, every parcel end to end) plus
``parcel_offsets.dat`` (int64, length n_parcels + 1). Parcel ``i`` of
``parcel_order.txt`` is ``data[offsets[i]:offsets[i + 1]]``.

Parcels are ~60 bytes each, so one file per parcel cost 128 bytes of .npy
header and a full 4K block apiece: 381 GB and 99M files for a 1188-subject
cohort, against 9.5 GB and 7k files this way.

Usage:
    python scripts/parcellation/export_masked_jacobian_vectors.py [options]

Parameters:
    --input-dir PATH          Batch directory containing Jacobian images.
    --rois-dir PATH           ROI mask root (default: outputs/rois).
    --jacobian PATH           Log-Jacobian image used in single-image mode.
    --output-dir PATH         Output root (default: outputs/jacobian_parcel_vectors).
    --label INT               Export one label only; otherwise export all labels.
    --n-parcels INT           Export this many masks per selected label.
    --sample-mode MODE        ``first`` or seeded ``random`` (default: first).
    --seed INT                Seed for random sampling (default: 7).
    --num-workers INT         Worker threads per subject-label batch (default: 1).

Examples:
    python scripts/parcellation/export_masked_jacobian_vectors.py
    python scripts/parcellation/export_masked_jacobian_vectors.py --label 49 --n-parcels 50 --sample-mode random --seed 7
    python scripts/parcellation/export_masked_jacobian_vectors.py --input-dir data/jacobians --num-workers 4
"""

from __future__ import annotations

"""
CLI usage:
    python scripts/parcellation/export_masked_jacobian_vectors.py
    python scripts/parcellation/export_masked_jacobian_vectors.py --label 10 --n-parcels 100
    python scripts/parcellation/export_masked_jacobian_vectors.py --label 49 --n-parcels 50 --sample-mode random --seed 7
    python scripts/parcellation/export_masked_jacobian_vectors.py --input-dir data/jacobians --label 10
    python scripts/parcellation/export_masked_jacobian_vectors.py --jacobian path/to/logJac.nii.gz
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_integrity import (
    atomic_write_raw,
    atomic_write_text,
    file_signature,
    load_completion,
    read_json,
    sha256_file,
    sha256_lines,
    signatures_match,
    validate_raw_vector,
    write_completion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROIS_DIR = PROJECT_ROOT / "outputs" / "rois"
DEFAULT_JACOBIAN_PATH = PROJECT_ROOT / "data" / "subToMNI_relative_logJac_MNI.nii.gz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "jacobian_parcel_vectors"
DEFAULT_ROI_MANIFEST_NAME = "atlas_manifest.json"
DIRECT_MEAN_FILENAME = "direct_mean.dat"
DIRECT_MEDIAN_FILENAME = "direct_median.dat"
PARCEL_ORDER_FILENAME = "parcel_order.txt"
VECTOR_DATA_FILENAME = "parcel_vectors.dat"
VECTOR_OFFSETS_FILENAME = "parcel_offsets.dat"
VECTOR_DTYPE = "float32"
OFFSET_DTYPE = "int64"

LABEL_NAMES = {
    3: "Left cerebral cortex",
    4: "Left lateral ventricle",
    5: "Left inferior lateral ventricle",
    8: "Left cerebellum cortex",
    10: "Left thalamus",
    11: "Left caudate",
    12: "Left putamen",
    13: "Left pallidum",
    16: "Brain-stem",
    17: "Left hippocampus",
    18: "Left amygdala",
    24: "CSF",
    26: "Left accumbens area",
    28: "Left ventral DC",
    43: "Right lateral ventricle",
    44: "Right inferior lateral ventricle",
    49: "Right thalamus",
    50: "Right caudate",
    51: "Right putamen",
    52: "Right pallidum",
    53: "Right hippocampus",
    54: "Right amygdala",
    58: "Right accumbens area",
    60: "Right ventral DC",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the Jacobian image once, extract masked parcel vectors, and save "
            "them as one concatenated memmap per subject. By default this exports all "
            "parcels from all available ROI label folders under --rois-dir."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help=(
            "Directory containing Jacobian images to batch process. "
            "When provided, the script exports parcel vectors for every "
            "matching Jacobian file in that folder."
        ),
    )
    parser.add_argument(
        "--rois-dir",
        type=Path,
        default=DEFAULT_ROIS_DIR,
        help=f"Directory containing parcel masks grouped by ROI label (default: {DEFAULT_ROIS_DIR}).",
    )
    parser.add_argument(
        "--jacobian",
        type=Path,
        default=DEFAULT_JACOBIAN_PATH,
        help=f"Path to the log-Jacobian determinant image (default: {DEFAULT_JACOBIAN_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where subject vector memmaps will be saved (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--label",
        type=int,
        help="Explicit ROI label to use instead of the default thalamus-first selection.",
    )
    parser.add_argument(
        "--n-parcels",
        type=int,
        default=None,
        help="Number of parcel vectors to export (default: all parcels available in the chosen label).",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("first", "random"),
        default="first",
        help="Use the first N sorted parcels or a seeded random sample (default: first).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used when --sample-mode random is selected.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker threads used to export parcel vectors "
            "within each subject and label batch (default: 1)."
        ),
    )
    parser.add_argument(
        "--roi-manifest",
        type=Path,
        default=None,
        help="Validated atlas manifest (default: <rois-dir>/atlas_manifest.json).",
    )
    parser.add_argument(
        "--allow-unvalidated-rois",
        action="store_true",
        help="Allow extraction without an atlas manifest (diagnostic use only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite every expected vector even if a matching completion manifest exists.",
    )
    return parser.parse_args()


def find_label_dirs(rois_dir: Path) -> list[Path]:
    label_dirs = sorted(
        (path for path in rois_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not label_dirs:
        raise ValueError(f"No ROI label directories found under {rois_dir}")
    return label_dirs


def process_jacobian_paths(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    patterns = (
        "subToMNI_relative_logJac_*.nii.gz",
        "subToMNI_relative_logJac_*.nii",
        "sub-*/sub-*_to_template_logJacobian.nii.gz",
        "sub-*/sub-*_to_template_logJacobian.nii",
    )
    jacobian_paths = sorted({path for pattern in patterns for path in input_dir.glob(pattern)})
    if not jacobian_paths:
        raise ValueError(f"No Jacobian images found in {input_dir}")
    return jacobian_paths


def strip_nii_suffix(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.suffix == ".nii":
        return path.stem
    return path.stem


def subject_id_from_jacobian_path(jacobian_path: Path) -> str:
    stem = strip_nii_suffix(jacobian_path)
    known_prefixes = (
        "subToMNI_relative_logJac_",
        "subToMNI_logJac_",
    )
    for prefix in known_prefixes:
        if stem.startswith(prefix):
            subject_id = stem[len(prefix):]
            if subject_id:
                return subject_id

    # Prefer the BIDS-like subject token when the Jacobian filename contains
    # additional processing suffixes such as "_to_template_logJacobian".
    for part in stem.split("_"):
        if part.startswith("sub-") and len(part) > 4:
            return part

    return stem

def count_parcels(label_dir: Path) -> int:
    return len(sorted(label_dir.glob("roi_*.nii.gz")))


def choose_label_dirs(rois_dir: Path, explicit_label: int | None, min_parcels: int | None) -> list[Path]:
    if explicit_label is not None:
        label_dir = rois_dir / str(explicit_label)
        if not label_dir.is_dir():
            raise ValueError(f"Requested label {explicit_label} is not available under {rois_dir}")
        total = count_parcels(label_dir)
        if min_parcels is not None and total < min_parcels:
            raise ValueError(
                f"Requested label {explicit_label} only has {total} parcels, cannot select {min_parcels}"
            )
        return [label_dir]

    label_dirs = [
        label_dir
        for label_dir in find_label_dirs(rois_dir)
        if count_parcels(label_dir) > 0 and (min_parcels is None or count_parcels(label_dir) >= min_parcels)
    ]
    if label_dirs:
        return label_dirs

    if min_parcels is None:
        raise ValueError(f"No ROI label directories found under {rois_dir}")
    raise ValueError(f"No ROI label contains at least {min_parcels} parcels under {rois_dir}")


def select_parcel_paths(label_dir: Path, n_parcels: int | None, sample_mode: str, seed: int) -> list[Path]:

    # Get all parcel paths and check if there are enough parcels to select from
    parcel_paths = sorted(
        label_dir.glob("roi_*.nii.gz"),
        key=lambda path: int(path.name[4:-7]),
    )
    if n_parcels is None:
        return parcel_paths

    if len(parcel_paths) < n_parcels:
        raise ValueError(
            f"Label {label_dir.name} only has {len(parcel_paths)} parcels, cannot select {n_parcels}"
        )

    # Select the first N parcels or a random sample of N parcels based on the sample_mode
    if sample_mode == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(parcel_paths, n_parcels))

    return parcel_paths[:n_parcels]


def extract_vectors(
    parcel_paths: list[Path],
    jacobian_img: nib.Nifti1Image,
    jacobian_data: np.ndarray,
    num_workers: int,
) -> list[np.ndarray]:
    """Return one float32 vector per parcel, in the order given."""

    def extract_single_parcel(parcel_path: Path) -> np.ndarray:
        # Load the parcel image and validate its shape and affine against the Jacobian image
        parcel_img = nib.load(str(parcel_path))
        if parcel_img.shape != jacobian_img.shape:
            raise ValueError(
                f"Shape mismatch for {parcel_path}: expected {jacobian_img.shape}, got {parcel_img.shape}"
            )
        if not np.allclose(parcel_img.affine, jacobian_img.affine):
            raise ValueError(f"Affine mismatch for {parcel_path}")

        # Extract the Jacobian values inside the mask, as float32 for compact storage
        vector = jacobian_data[parcel_img.get_fdata() > 0].astype(np.float32, copy=False)

        if vector.ndim != 1:
            raise ValueError(f"Expected a 1D masked vector for {parcel_path}, got shape {vector.shape}")
        if vector.size == 0:
            raise ValueError(f"Parcel mask contains no voxels: {parcel_path}")
        if not np.isfinite(vector).all():
            raise ValueError(f"Masked log-Jacobian vector contains non-finite values: {parcel_path}")
        return vector

    # executor.map preserves input order, so the results line up with parcel_paths.
    max_workers = min(num_workers, len(parcel_paths)) if parcel_paths else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(extract_single_parcel, parcel_paths))


def parcel_ids_for_exports(label_exports: list[tuple[int, list[Path]]]) -> list[str]:
    return [
        f"label_{label}/{parcel_path.stem.replace('.nii', '')}"
        for label, parcel_paths in label_exports
        for parcel_path in parcel_paths
    ]


def load_atlas_contract(args: argparse.Namespace, parcel_ids: list[str]) -> tuple[dict, str]:
    """Require either the legacy count contract or the validated connected-atlas contract."""
    manifest_path = args.roi_manifest or args.rois_dir / DEFAULT_ROI_MANIFEST_NAME
    if not manifest_path.is_file():
        if not args.allow_unvalidated_rois:
            raise ValueError(
                f"ROI count-check manifest not found: {manifest_path}. "
                "Run scripts/validation/check_roi_counts.py first."
            )
        return {"atlas_id": "UNVALIDATED", "parcel_count": len(parcel_ids)}, "UNVALIDATED"

    atlas = read_json(manifest_path)
    legacy_valid = atlas.get("all_ok") is True
    connected_valid = (
        atlas.get("schema_version") == 2
        and atlas.get("validation_status") == "passed"
        and atlas.get("labels_processed_sequentially") is True
    )
    if not (legacy_valid or connected_valid):
        bad_labels = sorted(
            row.get("label") for row in atlas.get("labels", []) if row.get("status") != "ok"
        )
        raise ValueError(
            f"ROI count check has not passed for labels {bad_labels}: {manifest_path}. "
            "Re-run scripts/validation/check_roi_counts.py."
        )
    manifest_hash = sha256_file(manifest_path)
    if connected_valid:
        expected_counts: dict[int, int] = {}
        for parcel_id in parcel_ids:
            label = int(parcel_id.split("/", 1)[0].removeprefix("label_"))
            expected_counts[label] = expected_counts.get(label, 0) + 1
        manifest_rows = {int(row["label"]): row for row in atlas.get("labels", [])}
        manifest_counts = {
            label: int(row["parcel_count"])
            for label, row in manifest_rows.items()
        }
        mismatches = {
            label: (count, manifest_counts.get(label))
            for label, count in expected_counts.items()
            if (
                count > manifest_counts.get(label, -1)
                or (args.n_parcels is None and count != manifest_counts.get(label))
            )
        }
        if mismatches:
            raise ValueError(
                f"Selected parcel counts do not match the validated atlas: {mismatches}"
            )

        for label in expected_counts:
            row = manifest_rows[label]
            label_manifest_path = args.rois_dir / str(label) / "label_manifest.json"
            if not label_manifest_path.is_file():
                raise ValueError(f"Missing label manifest: {label_manifest_path}")
            if sha256_file(label_manifest_path) != row.get("manifest_sha256"):
                raise ValueError(f"Label manifest hash mismatch: {label_manifest_path}")
            label_manifest = read_json(label_manifest_path)
            if (
                label_manifest.get("validation_status") != "passed"
                or int(label_manifest.get("label", -1)) != label
                or int(label_manifest.get("parcel_count", -1)) != manifest_counts[label]
            ):
                raise ValueError(f"Invalid label manifest: {label_manifest_path}")

        atlas.setdefault("atlas_id", manifest_hash)
    return atlas, manifest_hash


def subject_is_complete(
    subject_dir: Path,
    source_signature: dict,
    atlas_manifest_hash: str,
    parcel_ids: list[str],
) -> bool:
    completion = load_completion(subject_dir)
    if completion is None or completion.get("stage") != "parcel_vectors":
        return False
    if completion.get("atlas_manifest_sha256") != atlas_manifest_hash:
        return False
    if completion.get("parcel_order_sha256") != sha256_lines(parcel_ids):
        return False
    if not signatures_match(completion.get("source_jacobian", {}), source_signature):
        return False
    try:
        validate_raw_vector(subject_dir / DIRECT_MEAN_FILENAME, length=len(parcel_ids))
        validate_raw_vector(subject_dir / DIRECT_MEDIAN_FILENAME, length=len(parcel_ids))

        # The offsets index pins the expected size of the concatenated vector block.
        offsets = validate_raw_vector(
            subject_dir / VECTOR_OFFSETS_FILENAME, length=len(parcel_ids) + 1, dtype=OFFSET_DTYPE
        )
        expected_bytes = int(offsets[-1]) * np.dtype(VECTOR_DTYPE).itemsize
        if (subject_dir / VECTOR_DATA_FILENAME).stat().st_size != expected_bytes:
            return False
    except (OSError, ValueError):
        return False
    return True


def write_direct_summaries(subject_dir: Path, vectors: list[np.ndarray], parcel_ids: list[str]) -> None:
    # The vectors are already in memory, so no need to read back what we just wrote.
    means = np.array([np.mean(vector, dtype=np.float64) for vector in vectors], dtype=np.float64)
    medians = np.array([np.median(vector) for vector in vectors], dtype=np.float64)

    atomic_write_raw(subject_dir / DIRECT_MEAN_FILENAME, means, "float64")
    atomic_write_raw(subject_dir / DIRECT_MEDIAN_FILENAME, medians, "float64")
    atomic_write_text(subject_dir / PARCEL_ORDER_FILENAME, "\n".join(parcel_ids) + "\n")


def export_subject(
    *,
    subject_id: str,
    jacobian_path: Path,
    label_exports: list[tuple[int, list[Path]]],
    output_root: Path,
    num_workers: int,
    atlas: dict,
    atlas_manifest_hash: str,
    force: bool,
) -> None:
    subject_dir = output_root / subject_id
    parcel_ids = parcel_ids_for_exports(label_exports)
    source_signature = file_signature(jacobian_path)
    if not force and subject_is_complete(
        subject_dir, source_signature, atlas_manifest_hash, parcel_ids
    ):
        print(f"Validated parcel-vector output already complete: {subject_dir}", flush=True)
        return

    subject_dir.mkdir(parents=True, exist_ok=True)

    # Load the Jacobian once for the whole subject, not once per label directory.
    jacobian_img = nib.load(str(jacobian_path))
    jacobian_data = jacobian_img.get_fdata()
    print(f"Loaded Jacobian image once from {jacobian_path}", flush=True)
    print(f"Worker threads: {num_workers}", flush=True)

    vectors: list[np.ndarray] = []
    for label, parcel_paths in label_exports:
        vectors.extend(extract_vectors(parcel_paths, jacobian_img, jacobian_data, num_workers))
        print(f"[label {label:>3}] extracted {len(parcel_paths):6d} parcel vectors", flush=True)

    if not vectors:
        raise ValueError(f"No parcels selected for {subject_id}")

    # One concatenated float32 block plus an int64 index, rather than a file per parcel.
    offsets = np.cumsum([0] + [vector.size for vector in vectors], dtype=np.int64)
    atomic_write_raw(subject_dir / VECTOR_DATA_FILENAME, np.concatenate(vectors), VECTOR_DTYPE)
    atomic_write_raw(subject_dir / VECTOR_OFFSETS_FILENAME, offsets, OFFSET_DTYPE)

    final_source_signature = file_signature(jacobian_path)
    if not signatures_match(source_signature, final_source_signature):
        raise RuntimeError(f"Jacobian changed during extraction: {jacobian_path}")
    write_direct_summaries(subject_dir, vectors, parcel_ids)
    write_completion(subject_dir, {
        "schema_version": 2,
        "stage": "parcel_vectors",
        "subject_id": subject_id,
        "atlas_id": atlas.get("atlas_id"),
        "atlas_manifest_sha256": atlas_manifest_hash,
        "parcel_count": len(parcel_ids),
        "parcel_order_sha256": sha256_lines(parcel_ids),
        "source_jacobian": final_source_signature,
        "direct_summary_dtype": "float64",
        "direct_mean_file": DIRECT_MEAN_FILENAME,
        "direct_median_file": DIRECT_MEDIAN_FILENAME,
        "vector_data_file": VECTOR_DATA_FILENAME,
        "vector_offsets_file": VECTOR_OFFSETS_FILENAME,
        "vector_dtype": VECTOR_DTYPE,
        "offset_dtype": OFFSET_DTYPE,
    })
    print(f"Validated parcel-vector stage complete: {subject_dir}", flush=True)


def export_batch_vectors(
    jacobian_paths: list[Path],
    label_exports: list[tuple[int, list[Path]]],
    output_root: Path,
    num_workers: int,
    atlas: dict,
    atlas_manifest_hash: str,
    force: bool,
) -> None:
    total_subjects = len(jacobian_paths)

    for subject_index, jacobian_path in enumerate(jacobian_paths, start=1):
        subject_id = subject_id_from_jacobian_path(jacobian_path)
        print(
            f"[subject {subject_index:03d}/{total_subjects:03d}] exporting {subject_id} "
            f"from {jacobian_path}",
            flush=True,
        )
        export_subject(
            subject_id=subject_id,
            jacobian_path=jacobian_path,
            label_exports=label_exports,
            output_root=output_root,
            num_workers=num_workers,
            atlas=atlas,
            atlas_manifest_hash=atlas_manifest_hash,
            force=force,
        )


def main() -> None:
    args = parse_args()

    if args.n_parcels is not None and args.n_parcels < 1:
        raise ValueError("--n-parcels must be at least 1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")

    label_dirs = choose_label_dirs(args.rois_dir, args.label, args.n_parcels)
    label_exports: list[tuple[int, list[Path]]] = []
    for label_dir in label_dirs:
        label = int(label_dir.name)
        requested_count = args.n_parcels or count_parcels(label_dir)
        parcel_paths = select_parcel_paths(label_dir, requested_count, args.sample_mode, args.seed)
        label_exports.append((label, parcel_paths))

    parcel_ids = parcel_ids_for_exports(label_exports)
    atlas, atlas_manifest_hash = load_atlas_contract(args, parcel_ids)

    if args.label is not None:
        label_name = LABEL_NAMES.get(args.label, "Unknown label")
        print(f"Selected label {args.label} ({label_name})", flush=True)
    else:
        print(f"Selected all ROI labels under {args.rois_dir}", flush=True)
        print(f"Label count: {len(label_exports)}", flush=True)

    print(f"Sample mode: {args.sample_mode}", flush=True)
    if args.n_parcels is None:
        print("Parcel count requested: all parcels in each selected label", flush=True)
    else:
        print(f"Parcel count requested: {args.n_parcels} parcels per selected label", flush=True)
    if args.sample_mode == "random":
        print(f"Random seed: {args.seed}", flush=True)
    print(f"Worker threads requested: {args.num_workers}", flush=True)

    if args.input_dir is not None:
        jacobian_paths = process_jacobian_paths(args.input_dir)
        print(f"Batch mode: found {len(jacobian_paths)} Jacobian images in {args.input_dir}", flush=True)
        export_batch_vectors(
            jacobian_paths=jacobian_paths,
            label_exports=label_exports,
            output_root=args.output_dir,
            num_workers=args.num_workers,
            atlas=atlas,
            atlas_manifest_hash=atlas_manifest_hash,
            force=args.force,
        )
        return

    subject_id = subject_id_from_jacobian_path(args.jacobian)
    print(f"Single-image mode subject ID: {subject_id}", flush=True)
    export_subject(
        subject_id=subject_id,
        jacobian_path=args.jacobian,
        label_exports=label_exports,
        output_root=args.output_dir,
        num_workers=args.num_workers,
        atlas=atlas,
        atlas_manifest_hash=atlas_manifest_hash,
        force=args.force,
    )


if __name__ == "__main__":
    main()
