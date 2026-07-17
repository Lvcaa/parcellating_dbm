"""
Run ANTs nonlinear registration and compute a Jacobian determinant image.

ANTs must be installed with ``antsRegistrationSyNQuick.sh`` and
``CreateJacobianDeterminantImage`` available in ``PATH``. Outputs are written
under ``<output-root>/<subject-id>/``.

When a forward warp field has already been computed elsewhere (e.g. copied in
under data/warps/<subject-id>/Reg_/_SyN1Warp.nii.gz), pass it via
--warp-image to skip registration entirely and go straight to the Jacobian
computation.

Usage:
    python scripts/registration/run_ants_jacobian.py --fixed-image PATH [options]
    python scripts/registration/run_ants_jacobian.py --warp-image PATH --subject-id ID

Parameters:
    --fixed-image PATH       Reference image in target space. Required unless --warp-image is given.
    --moving-image PATH      Subject T1 image to register.
    --warp-image PATH        Precomputed forward warp field; skips registration when given.
    --output-root PATH       ANTs output root (default: outputs/ants_registration).
    --subject-id TEXT        Output-folder ID; required with --warp-image, otherwise inferred.
    --dimension {2,3}        Image dimensionality (default: 3).
    --transform-type TYPE    ANTs transform type: t, r, a, s, sr, so, or b.
    --threads INT            ITK thread count (default: 1).
    --write-raw-jacobian     Also write the raw Jacobian image.

Examples:
    python scripts/registration/run_ants_jacobian.py --fixed-image data/reference/MNI152_T1_1mm.nii.gz
    python scripts/registration/run_ants_jacobian.py --fixed-image data/reference/MNI152_T1_1mm.nii.gz --moving-image data/subjects/sub-0006_T1w.nii.gz --subject-id sub-0006 --threads 8 --write-raw-jacobian
    python scripts/registration/run_ants_jacobian.py --warp-image data/warps/sub-0006/Reg_/_SyN1Warp.nii.gz --subject-id sub-0006
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_integrity import file_signature, load_completion, signatures_match, write_completion


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
        default=None,
        help=(
            "Reference image in target space, typically an MNI T1 template. "
            "Required unless --warp-image is provided."
        ),
    )
    parser.add_argument(
        "--moving-image",
        type=Path,
        default=DEFAULT_MOVING_IMAGE,
        help=f"Subject T1 image to register (default: {DEFAULT_MOVING_IMAGE}).",
    )
    parser.add_argument(
        "--warp-image",
        type=Path,
        default=None,
        help=(
            "Path to an already-computed forward warp field (e.g. an ANTs "
            "_SyN1Warp.nii.gz). When provided, registration is skipped entirely "
            "and this warp is used directly to compute the Jacobian; "
            "--subject-id is then required and --fixed-image/--moving-image are ignored."
        ),
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
    parser.add_argument(
        "--jacobian-geometric",
        choices=("true", "false"),
        default="false",
        help="ANTs useGeometric flag; both modes produce relative log-Jacobians (default: false).",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate outputs despite a valid marker.")
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


def validate_scalar_nifti(path: Path) -> None:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3-D Jacobian image, got {image.shape}: {path}")
    values = np.asanyarray(image.dataobj)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite Jacobian values: {path}")


def jacobian_is_complete(
    output_dir: Path,
    source_warp: dict,
    log_path: Path,
    raw_path: Path,
    write_raw: bool,
    geometric: bool,
) -> bool:
    completion = load_completion(output_dir)
    if completion is None or completion.get("stage") != "log_jacobian":
        return False
    if completion.get("jacobian_geometric") != geometric:
        return False
    if completion.get("raw_jacobian_saved") != write_raw:
        return False
    if not signatures_match(completion.get("source_warp", {}), source_warp):
        return False
    try:
        validate_scalar_nifti(log_path)
        if write_raw:
            validate_scalar_nifti(raw_path)
    except (OSError, ValueError):
        return False
    return True


def main() -> None:
    args = parse_args()

    if args.warp_image is not None and not args.subject_id:
        raise ValueError("--subject-id is required when --warp-image is provided")
    if args.warp_image is None and args.fixed_image is None:
        raise ValueError("--fixed-image is required unless --warp-image is provided")

    subject_id = args.subject_id or infer_subject_id(args.moving_image)
    subject_output_dir = args.output_root / subject_id

    output_prefix = subject_output_dir / f"{subject_id}_to_template_"
    warp_path = (
        args.warp_image
        if args.warp_image is not None
        else subject_output_dir / f"{subject_id}_to_template_1Warp.nii.gz"
    )
    log_jacobian_path = subject_output_dir / f"{subject_id}_to_template_logJacobian.nii.gz"
    raw_jacobian_path = subject_output_dir / f"{subject_id}_to_template_jacobian.nii.gz"

    if args.warp_image is not None:
        validate_input_image(args.warp_image, "Warp image")
    else:
        validate_input_image(args.fixed_image, "Fixed image")
        validate_input_image(args.moving_image, "Moving image")
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)

    if warp_path.is_file():
        print(f"Warp already exists; skipping registration: {warp_path}", flush=True)
    else:
        ants_registration = resolve_executable("antsRegistrationSyNQuick.sh")
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
        raise FileNotFoundError(f"Expected ANTs warp was not created: {warp_path}")

    geometric = args.jacobian_geometric == "true"
    source_warp = file_signature(warp_path)
    if not args.force and jacobian_is_complete(
        subject_output_dir,
        source_warp,
        log_jacobian_path,
        raw_jacobian_path,
        args.write_raw_jacobian,
        geometric,
    ):
        print(f"Validated Jacobian output already complete: {subject_output_dir}", flush=True)
        return

    jacobian_tool = resolve_executable("CreateJacobianDeterminantImage")
    geometric_flag = "1" if geometric else "0"
    temporary_log = subject_output_dir / f".{subject_id}.{os.getpid()}.tmp.logJacobian.nii.gz"
    temporary_raw = subject_output_dir / f".{subject_id}.{os.getpid()}.tmp.jacobian.nii.gz"
    try:
        log_jacobian_command = [
            jacobian_tool,
            str(args.dimension),
            str(warp_path),
            str(temporary_log),
            "1",
            geometric_flag,
        ]
        run_command(log_jacobian_command, env)
        validate_scalar_nifti(temporary_log)
        os.replace(temporary_log, log_jacobian_path)

        if args.write_raw_jacobian:
            raw_jacobian_command = [
                jacobian_tool,
                str(args.dimension),
                str(warp_path),
                str(temporary_raw),
                "0",
                geometric_flag,
            ]
            run_command(raw_jacobian_command, env)
            validate_scalar_nifti(temporary_raw)
            os.replace(temporary_raw, raw_jacobian_path)
    finally:
        temporary_log.unlink(missing_ok=True)
        temporary_raw.unlink(missing_ok=True)

    final_source_warp = file_signature(warp_path)
    if not signatures_match(source_warp, final_source_warp):
        raise RuntimeError(f"Warp changed during Jacobian generation: {warp_path}")
    write_completion(subject_output_dir, {
        "schema_version": 1,
        "stage": "log_jacobian",
        "subject_id": subject_id,
        "source_warp": final_source_warp,
        "warp_direction": "subject_to_template",
        "relative_jacobian": True,
        "log_jacobian": True,
        "jacobian_geometric": geometric,
        "raw_jacobian_saved": bool(args.write_raw_jacobian),
        "dimension": int(args.dimension),
        "log_jacobian_file": log_jacobian_path.name,
    })

    print(f"Registration output folder: {subject_output_dir}", flush=True)
    print(f"Warp field: {warp_path}", flush=True)
    print(f"Log-Jacobian image: {log_jacobian_path}", flush=True)
    if args.write_raw_jacobian:
        print(f"Raw Jacobian image: {raw_jacobian_path}", flush=True)


if __name__ == "__main__":
    main()
