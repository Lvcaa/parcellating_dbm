"""
Compute log-Jacobian determinant images for all subjects in DATASET_DIR.

For each subject the script looks for an existing warp (_SyN1Warp.nii.gz).
If the warp is present, registration is skipped and only the log-Jacobian is
computed. If the warp is missing and --fixed-image is provided, a full ANTs
registration is run first.  Subjects whose log-Jacobian already exists are
skipped entirely.

ANTs must be installed with ``antsRegistrationSyNQuick.sh`` and
``CreateJacobianDeterminantImage`` available in PATH.

Usage:
    python scripts/registration/run_ants_jacobian.py [options]

Parameters:
    --fixed-image PATH   MNI reference image (required only when warp is missing).
    --threads INT        ITK thread count (default: 1).
    --dimension {2,3}    Image dimensionality (default: 3).
    --transform-type     ANTs transform type (default: s).

Examples:
    python scripts/registration/run_ants_jacobian.py
    python scripts/registration/run_ants_jacobian.py --threads 8
    python scripts/registration/run_ants_jacobian.py --fixed-image data/reference/MNI152_T1_1mm.nii.gz --threads 8
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import const


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute log-Jacobian images for all subjects in DATASET_DIR. "
            "Skips subjects whose log-Jacobian already exists. "
            "Skips registration when the warp is already present."
        )
    )
    parser.add_argument(
        "--fixed-image",
        type=Path,
        default=None,
        help="MNI reference image used for registration (only needed if warp is missing).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of ITK threads (default: 1).",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=3,
        choices=(2, 3),
        help="Image dimensionality (default: 3).",
    )
    parser.add_argument(
        "--transform-type",
        type=str,
        default="s",
        choices=("t", "r", "a", "s", "sr", "so", "b"),
        help="ANTs transform type (default: s).",
    )
    return parser.parse_args()


def resolve_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(
            f"Required executable not found in PATH: {name}. "
            "Make sure ANTs is installed and available in your shell."
        )
    return executable


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def run_registration(
    subject_dir: Path,
    fixed_image: Path,
    moving_image: Path,
    transform_type: str,
    dimension: int,
    env: dict[str, str],
) -> Path:
    """Run ANTs registration and return the warp path."""
    ants_registration = resolve_executable("antsRegistrationSyNQuick.sh")
    output_prefix = subject_dir / "_SyN"
    warp_path = subject_dir / WARP_FILENAME
    run_command(
        [
            ants_registration,
            "-d", str(dimension),
            "-f", str(fixed_image),
            "-m", str(moving_image),
            "-t", transform_type,
            "-o", str(output_prefix),
        ],
        env,
    )
    if not warp_path.is_file():
        raise FileNotFoundError(f"ANTs warp not produced: {warp_path}")
    return warp_path


def compute_log_jacobian(
    warp_path: Path,
    log_jacobian_path: Path,
    dimension: int,
    env: dict[str, str],
) -> None:
    """Compute the log-Jacobian determinant from a warp field."""
    jacobian_tool = resolve_executable("CreateJacobianDeterminantImage")
    # "1" = log transform, "0" = geometric only (excludes affine component)
    run_command(
        [
            jacobian_tool,
            str(dimension),
            str(warp_path),
            str(log_jacobian_path),
            "1",
            "0",
        ],
        env,
    )


def process_subject(
    subject_dir: Path,
    args: argparse.Namespace,
    env: dict[str, str],
) -> str:
    """
    Process one subject. Returns a status string: 'skipped', 'jacobian_only', or 'full'.
    """
    log_jacobian_path = subject_dir / LOG_JACOBIAN_FILENAME
    warp_path         = subject_dir / WARP_FILENAME

    if log_jacobian_path.exists():
        return "skipped"

    if not warp_path.exists():
        if args.fixed_image is None:
            print(
                f"  [WARNING] Warp missing and --fixed-image not provided, skipping.",
                flush=True,
            )
            return "skipped"
        moving_image = subject_dir / f"{subject_dir.name}_T1w.nii.gz"
        if not moving_image.is_file():
            print(
                f"  [WARNING] Warp missing and no T1w found at {moving_image}, skipping.",
                flush=True,
            )
            return "skipped"
        warp_path = run_registration(
            subject_dir, args.fixed_image, moving_image,
            args.transform_type, args.dimension, env,
        )
        compute_log_jacobian(warp_path, log_jacobian_path, args.dimension, env)
        return "full"

    compute_log_jacobian(warp_path, log_jacobian_path, args.dimension, env)
    return "jacobian_only"


def main() -> None:
    args = parse_args()

    if args.fixed_image is not None and not args.fixed_image.is_file():
        raise FileNotFoundError(f"Fixed image not found: {args.fixed_image}")

    env = dict(os.environ)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)

    subject_dirs = sorted(
        d for d in const.DATASET_DIR.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    total = len(subject_dirs)
    counts = {"skipped": 0, "jacobian_only": 0, "full": 0}

    for i, subject_dir in enumerate(subject_dirs, start=1):
        print(f"[{i:04d}/{total}] {subject_dir.name}", flush=True)
        status = process_subject(subject_dir, args, env)
        counts[status] += 1
        print(f"  -> {status}", flush=True)

    print(
        f"\nDone. jacobian_only={counts['jacobian_only']}  "
        f"full={counts['full']}  skipped={counts['skipped']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
