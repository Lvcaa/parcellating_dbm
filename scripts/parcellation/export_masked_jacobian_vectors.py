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
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROIS_DIR = PROJECT_ROOT / "outputs" / "rois"
DEFAULT_JACOBIAN_PATH = PROJECT_ROOT / "data" / "subToMNI_relative_logJac_MNI.nii.gz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "jacobian_parcel_vectors"

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
            "one .npy file per parcel. By default this exports all parcels from all "
            "available ROI label folders under --rois-dir."
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
        help=f"Path to the Jacobian determinant image (default: {DEFAULT_JACOBIAN_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where .npy vectors will be saved (default: {DEFAULT_OUTPUT_DIR}).",
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
    return parser.parse_args()


def find_label_dirs(rois_dir: Path) -> list[Path]:
    label_dirs = sorted(path for path in rois_dir.iterdir() if path.is_dir())
    if not label_dirs:
        raise ValueError(f"No ROI label directories found under {rois_dir}")
    return label_dirs


def process_jacobian_paths(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    jacobian_paths = sorted(input_dir.glob("subToMNI_relative_logJac_*.nii.gz"))
    if not jacobian_paths:
        jacobian_paths = sorted(input_dir.glob("subToMNI_relative_logJac_*.nii"))
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
    parcel_paths = sorted(label_dir.glob("roi_*.nii.gz"))
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


def export_vectors(
    parcel_paths: list[Path],
    jacobian_path: Path,
    output_dir: Path,
    num_workers: int,
) -> None:

    # Load the Jacobian image once and keep it in memory for all parcel processing
    jacobian_img = nib.load(str(jacobian_path))

    # Preload the Jacobian data into memory to avoid repeated disk access during masking
    jacobian_data = jacobian_img.get_fdata()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded Jacobian image once from {jacobian_path}", flush=True)
    print(f"Writing .npy vectors to {output_dir}", flush=True)
    print(f"Worker threads: {num_workers}", flush=True)

    def export_single_parcel(index_and_path: tuple[int, Path]) -> tuple[int, str, tuple[int, ...]]:
        index, parcel_path = index_and_path

        # Load the parcel image and validate its shape and affine against the Jacobian image
        parcel_img = nib.load(str(parcel_path))
        if parcel_img.shape != jacobian_img.shape:
            raise ValueError(
                f"Shape mismatch for {parcel_path}: expected {jacobian_img.shape}, got {parcel_img.shape}"
            )
        if not np.allclose(parcel_img.affine, jacobian_img.affine):
            raise ValueError(f"Affine mismatch for {parcel_path}")

        # Create a boolean mask where the parcel image has values greater than 0
        mask = parcel_img.get_fdata() > 0

        # Extract the Jacobian values at the masked locations and convert to float32 for efficient storage
        vector = jacobian_data[mask].astype(np.float32, copy=False)

        if vector.ndim != 1:
            raise ValueError(f"Expected a 1D masked vector for {parcel_path}, got shape {vector.shape}")

        # Save the extracted vector to a .npy file named after the parcel
        output_path = output_dir / f"{parcel_path.stem.replace('.nii', '')}.npy"
        np.save(output_path, vector)
        return index, output_path.name, vector.shape

    indexed_paths = list(enumerate(parcel_paths, start=1))
    max_workers = min(num_workers, len(indexed_paths)) if indexed_paths else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, output_name, vector_shape in executor.map(export_single_parcel, indexed_paths):
            print(
                f"[{index:03d}/{len(parcel_paths):03d}] saved {output_name} with shape {vector_shape}",
                flush=True,
            )


def export_batch_vectors(
    jacobian_paths: list[Path],
    label_exports: list[tuple[int, list[Path]]],
    output_root: Path,
    num_workers: int,
) -> None:
    total_subjects = len(jacobian_paths)

    for subject_index, jacobian_path in enumerate(jacobian_paths, start=1):
        subject_id = subject_id_from_jacobian_path(jacobian_path)
        print(
            f"[subject {subject_index:03d}/{total_subjects:03d}] exporting {subject_id} "
            f"from {jacobian_path}",
            flush=True,
        )
        for label, parcel_paths in label_exports:
            subject_output_dir = output_root / subject_id / f"label_{label}"
            export_vectors(
                parcel_paths=parcel_paths,
                jacobian_path=jacobian_path,
                output_dir=subject_output_dir,
                num_workers=num_workers,
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
        )
        return

    subject_id = subject_id_from_jacobian_path(args.jacobian)
    print(f"Single-image mode subject ID: {subject_id}", flush=True)
    for label, parcel_paths in label_exports:
        output_dir = args.output_dir / subject_id / f"label_{label}"
        export_vectors(
            parcel_paths=parcel_paths,
            jacobian_path=args.jacobian,
            output_dir=output_dir,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
