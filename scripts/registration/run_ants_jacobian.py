from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOVING_IMAGE = PROJECT_ROOT / "data" / "reference" / "sub-0091_ses-V01_T1w.nii.gz"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ants_registration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ANTs nonlinear registration for one T1 image and compute a "
            "Jacobian determinant image from the resulting warp."
        )
    )
    parser.add_argument(
        "--fixed-image",
        type=Path,
        required=True,
        help="Reference image in target space, typically an MNI T1 template.",
    )
    parser.add_argument(
        "--moving-image",
        type=Path,
        default=DEFAULT_MOVING_IMAGE,
        help=f"Subject T1 image to register (default: {DEFAULT_MOVING_IMAGE}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory where ANTs outputs will be written (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Optional subject identifier used in the output folder name.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=3,
        choices=(2, 3),
        help="Image dimensionality passed to ANTs (default: 3).",
    )
    parser.add_argument(
        "--transform-type",
        type=str,
        default="s",
        choices=("t", "r", "a", "s", "sr", "so", "b"),
        help="Transform type for antsRegistrationSyNQuick.sh (default: s).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of ITK threads to use (default: 1).",
    )
    parser.add_argument(
        "--write-raw-jacobian",
        action="store_true",
        help="Also write the raw Jacobian determinant image in addition to log-Jacobian.",
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


def validate_input_image(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def infer_subject_id(moving_image: Path) -> str:
    name = moving_image.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if moving_image.suffix == ".nii":
        return moving_image.stem
    return moving_image.stem


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    args = parse_args()

    validate_input_image(args.fixed_image, "Fixed image")
    validate_input_image(args.moving_image, "Moving image")

    ants_registration = resolve_executable("antsRegistrationSyNQuick.sh")
    jacobian_tool = resolve_executable("CreateJacobianDeterminantImage")

    subject_id = args.subject_id or infer_subject_id(args.moving_image)
    subject_output_dir = args.output_root / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = subject_output_dir / f"{subject_id}_to_template_"
    warp_path = subject_output_dir / f"{subject_id}_to_template_1Warp.nii.gz"
    log_jacobian_path = subject_output_dir / f"{subject_id}_to_template_logJacobian.nii.gz"
    raw_jacobian_path = subject_output_dir / f"{subject_id}_to_template_jacobian.nii.gz"

    env = dict(os.environ)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)

    registration_command = [
        ants_registration,
        "-d",
        str(args.dimension),
        "-f",
        str(args.fixed_image),
        "-m",
        str(args.moving_image),
        "-t",
        args.transform_type,
        "-o",
        str(output_prefix),
    ]
    run_command(registration_command, env)

    if not warp_path.is_file():
        raise FileNotFoundError(
            f"Expected ANTs warp was not created: {warp_path}"
        )

    log_jacobian_command = [
        jacobian_tool,
        str(args.dimension),
        str(warp_path),
        str(log_jacobian_path),
        "1",
        "0",
    ]
    run_command(log_jacobian_command, env)

    if args.write_raw_jacobian:
        raw_jacobian_command = [
            jacobian_tool,
            str(args.dimension),
            str(warp_path),
            str(raw_jacobian_path),
            "0",
            "0",
        ]
        run_command(raw_jacobian_command, env)

    print(f"Registration output folder: {subject_output_dir}", flush=True)
    print(f"Warp field: {warp_path}", flush=True)
    print(f"Log-Jacobian image: {log_jacobian_path}", flush=True)
    if args.write_raw_jacobian:
        print(f"Raw Jacobian image: {raw_jacobian_path}", flush=True)


if __name__ == "__main__":
    main()
