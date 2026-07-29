"""
Compute a log-Jacobian determinant image from an already-registered warp field.

No registration happens here — this only wraps ANTs' CreateJacobianDeterminantImage
over a forward warp field that already exists (e.g. under
data/warps/<subject-id>/Reg_/_SyN1Warp.nii.gz). Outputs are written under
<output-root>/<subject-id>/.

Usage:
    python scripts/registration/compute_log_jacobian.py --warp-image PATH --subject-id ID [options]

Parameters:
    --warp-image PATH        Required. Precomputed forward warp field.
    --subject-id TEXT        Required. Output-folder identifier.
    --output-root PATH       ANTs output root (default: outputs/ants_registration).
    --dimension {2,3}        Image dimensionality (default: 3).
    --threads INT            ITK thread count (default: 1).
    --write-raw-jacobian     Also write the raw Jacobian image.
    --jacobian-geometric {true,false}  ANTs useGeometric flag (default: false).
    --force                  Regenerate outputs despite a valid completion marker.

Examples:
    python scripts/registration/compute_log_jacobian.py --warp-image data/warps/sub-0006/Reg_/_SyN1Warp.nii.gz --subject-id sub-0006
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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ants_registration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a log-Jacobian determinant image from an existing warp field."
    )
    parser.add_argument("--warp-image", type=Path, required=True, help="Precomputed forward warp field.")
    parser.add_argument("--subject-id", type=str, required=True, help="Output-folder identifier.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory where outputs will be written (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=3,
        choices=(2, 3),
        help="Image dimensionality passed to CreateJacobianDeterminantImage (default: 3).",
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
    if not args.warp_image.is_file():
        raise FileNotFoundError(f"Warp image does not exist: {args.warp_image}")

    subject_output_dir = args.output_root / args.subject_id
    log_jacobian_path = subject_output_dir / f"{args.subject_id}_to_template_logJacobian.nii.gz"
    raw_jacobian_path = subject_output_dir / f"{args.subject_id}_to_template_jacobian.nii.gz"
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)

    geometric = args.jacobian_geometric == "true"
    source_warp = file_signature(args.warp_image)
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
    temporary_log = subject_output_dir / f".{args.subject_id}.{os.getpid()}.tmp.logJacobian.nii.gz"
    temporary_raw = subject_output_dir / f".{args.subject_id}.{os.getpid()}.tmp.jacobian.nii.gz"
    try:
        log_jacobian_command = [
            jacobian_tool,
            str(args.dimension),
            str(args.warp_image),
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
                str(args.warp_image),
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

    final_source_warp = file_signature(args.warp_image)
    if not signatures_match(source_warp, final_source_warp):
        raise RuntimeError(f"Warp changed during Jacobian generation: {args.warp_image}")
    write_completion(subject_output_dir, {
        "schema_version": 1,
        "stage": "log_jacobian",
        "subject_id": args.subject_id,
        "source_warp": final_source_warp,
        "warp_direction": "subject_to_template",
        "relative_jacobian": True,
        "log_jacobian": True,
        "jacobian_geometric": geometric,
        "raw_jacobian_saved": bool(args.write_raw_jacobian),
        "dimension": int(args.dimension),
        "log_jacobian_file": log_jacobian_path.name,
    })

    print(f"Output folder: {subject_output_dir}", flush=True)
    print(f"Warp field: {args.warp_image}", flush=True)
    print(f"Log-Jacobian image: {log_jacobian_path}", flush=True)
    if args.write_raw_jacobian:
        print(f"Raw Jacobian image: {raw_jacobian_path}", flush=True)


if __name__ == "__main__":
    main()
